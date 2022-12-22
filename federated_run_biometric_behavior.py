from distutils.command.build import build
import pandas as pd
import math
from multiprocessing import Process
import time
import argparse
import os
import numpy as np
import tensorflow as tf 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils import check_random_state
import flwr as fl
import os
from sklearn.model_selection import train_test_split
from sktime_dl.classification import InceptionTimeClassifier, CNNClassifier, ResNetClassifier, \
    MCDCNNClassifier
from sktime_dl.utils import check_and_clean_data, \
    check_and_clean_validation_data
from typing import Any, Callable, Dict, List, Optional, Tuple
from sklearn.metrics import f1_score, precision_score , recall_score, multilabel_confusion_matrix
from flwr.common import (
    EvaluateRes,
    Scalar,
)

parser = argparse.ArgumentParser()
parser.add_argument('--sensor', type=str, default='gyro')
parser.add_argument('--rounds', type=int, default=10)
parser.add_argument('--batch', type=int, default=32)
parser.add_argument('--epoch', type=int, default=5)
parser.add_argument('--strategy', type=str, default='avg')
parser.add_argument('--nn', type=str, default='resnet')

args = parser.parse_args()

if args.nn == 'inception':
    network = InceptionTimeClassifier()
elif args.nn == 'cnn':
    network = CNNClassifier()
elif args.nn =='resnet':
    network = ResNetClassifier()
elif args.nn == 'mdcnn':
    network = MCDCNNClassifier()

batch = args.batch
epoch = args.epoch

NUM_CLIENTS = 60

if args.sensor == 'gyro':
    sensor_dir = 'gyroscope'
elif args.sensor == 'acc':
    sensor_dir = 'accelerometer'

path_server_aggregation_eval = f'results/distributed/{sensor_dir}/aggregation_eva_{batch}batch_{epoch}epoch.txt'
path_server_aggregation_fit = f'results/distributed/{sensor_dir}/aggregation_fit_{batch}batch_{epoch}epoch.txt'

header_eval = 'loss,accuracy,recall,precision,fpr,frr,f1,strategy,nn\n'
header_fit = 'loss,accuracy,strategy,cnn\n'

if not os.path.exists(path_server_aggregation_eval):
    with open(path_server_aggregation_eval, 'a') as file:
        file.write(header_eval)
if not os.path.exists(path_server_aggregation_fit):
    with open(path_server_aggregation_fit, 'a') as file:
        file.write(header_fit)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def fpr_micro(y_test, y_pred):
    mcm = multilabel_confusion_matrix(y_test, y_pred)
    tn = mcm[:, 0, 0].sum()
    tp = mcm[:, 1, 1].sum()
    fn = mcm[:, 1, 0].sum()
    fp = mcm[:, 0, 1].sum()
    fpr_micro = fp/(fp+tn)
    return fpr_micro

def frr_micro(y_test, y_pred):
    mcm = multilabel_confusion_matrix(y_test, y_pred)
    tn = mcm[:, 0, 0].sum()
    tp = mcm[:, 1, 1].sum()
    fn = mcm[:, 1, 0].sum()
    fp = mcm[:, 0, 1].sum()
    frr_micro = fn / (fn + tp)
    return frr_micro

def start_server():
    if args.strategy == 'avg':
        strategy = customFedAvg(
            fraction_eval=0.05,
            fraction_fit=0.1,  # Sample 10% of available clients for the next round
            min_fit_clients=10,  # Minimum number of clients to be sampled for the next round
            min_available_clients=int(10),
        )
    elif args.strategy == 'yogi':
        model = network.build_model(input_shape=(150,6), nb_classes=NUM_CLIENTS)
        strategy = fl.server.strategy.FedYogi(
            fraction_eval=0.05,
            fraction_fit=0.1,  # Sample 10% of available clients for the next round
            min_fit_clients=10,  # Minimum number of clients to be sampled for the next round
            min_available_clients=int(10),
            initial_parameters=fl.common.weights_to_parameters(model.get_weights(),)
        )
    elif args.strategy == 'adagrad':
        model = network.build_model(input_shape=(150,6), nb_classes=NUM_CLIENTS)
        strategy = CustomFedAdagrad(
            fraction_eval=0.05,
            fraction_fit=0.1,  # Sample 10% of available clients for the next round
            min_fit_clients=10,  # Minimum number of clients to be sampled for the next round
            min_available_clients=int(10),
            initial_parameters=fl.common.weights_to_parameters(model.get_weights()),
        )

    fl.server.start_server(server_address="localhost:8080", config={"num_rounds": args.rounds}, strategy=strategy)

def input_preprocessing(X, y):
        
    encoder = OneHotEncoder()

    X_new = check_and_clean_data(X, y, input_checks=True)
    
    return X_new

def janela_decisao(t,lq,ld,partition):
    proporcao = 1/lq
    ldx = math.floor((t)/(partition*lq))

    if ldx > lq:
        ldx = lq

    return ldx

def particiona(train_dataframe , partition):    
    #Dataset com a quantidade de dados em cada label;
    labels_quantity_proportion =  train_dataframe.label.value_counts()
    
    #Tamanho do dataset
    t = len(train_dataframe)

    # Quantidade dados na labels
    ld = labels_quantity_proportion[0]
    
    # Quantidade de labels
    lq = round(t/ld)
   
    #Cálculo da quantidade de dados que será pego de cada label para que a proporção das partições seja a mesma da do dataset principal 
    prop = janela_decisao(t,lq,ld,partition)  
    
    janela_i = 0
    janela_j = prop
    lista_particoes = []
    df_group = pd.DataFrame(pd.DataFrame(train_dataframe.groupby('label')))
  
    for x in range(0 ,partition):
    	
        df_concat = pd.DataFrame()
        
        for i in range(0, lq):
            
            df2_concat = df_group.iloc[i,1].iloc[janela_i:janela_j]
            df_concat = pd.concat([df_concat,df2_concat],axis = 0)
        
        lista_particoes.append(df_concat)
        
        janela_i = janela_i + prop
        janela_j = janela_j + prop
    return lista_particoes

def create_client(cid, train_dataframe_user, test_dataframe_user):
    print(cid)
    print("Running client ")
    

    # Define Flower client
    class TimeSeriesClient(fl.client.NumPyClient):
        def __init__(self, cid,model, x_train, y_train, x_test, y_test, x_val, y_val) -> None:
            self.cid = cid
            self.model = model
            self.x_train, self.y_train = x_train, y_train
            self.x_test, self.y_test = x_test, y_test
            self.x_val, self.y_val = x_val, y_val
        def get_parameters(self):
            return self.model.get_weights()
        def fit(self, parameters, config):
            learning_rate = 0.0001
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=50,
                                                                    min_lr=learning_rate)
            callbacks = [reduce_lr]
            self.model.set_weights(parameters)
            history = self.model.fit(self.x_train, self.y_train, verbose=1, batch_size=batch,epochs=epoch, 
                                     callbacks=callbacks,validation_data=(self.x_val, self.y_val))
           
            return self.model.get_weights(), len(self.x_train), {
                "accuracy": history.history["accuracy"][0],
                "loss": history.history["loss"][0]
            }
        def evaluate(self, parameters, config):
            # set shared weights in edge model
            self.model.set_weights(parameters)
            # evalute model
            loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=1)
            y_pred = self.model.predict(self.x_test)
            y_pred_metrics = np.argmax(y_pred, axis=1)
            y_test_metrics = np.argmax(self.y_test, axis=1)
            fpr = fpr_micro(y_test_metrics, y_pred_metrics)
            frr = frr_micro(y_test_metrics, y_pred_metrics)
            recall = recall_score(y_test_metrics, y_pred_metrics, average='micro')
            precision = precision_score(y_test_metrics, y_pred_metrics, average='micro')
            f1 = f1_score(y_test_metrics, y_pred_metrics, average='micro')
            return loss, len(self.x_test), {
                "accuracy": accuracy, 
                "loss": loss,
                "recall": recall,
                "precision": precision,
                "fpr": fpr,
                "frr": frr,
                "f1_score": f1,
            }

    X = train_dataframe_user.iloc[:, :-2]
    X_test = test_dataframe_user.iloc[:, :-2]
    
    y = np.vstack(train_dataframe_user['encoded_labels'].values)
    y_test = np.vstack(test_dataframe_user['encoded_labels'].values)
    
    X_train_new = pd.DataFrame(format_rows(X))
    X_test_new = pd.DataFrame(format_rows(X_test))
    
    X_train, X_val, y_train, y_val = train_test_split(X_train_new, y, test_size=0.33, random_state=42)

    x_train = input_preprocessing(X_train, y_train)
    x_val = input_preprocessing(X_val, y_val)
    x_test = input_preprocessing(X_test_new, y_test)
       
    model = network.build_model(input_shape=(200,3), nb_classes=NUM_CLIENTS)
   
    client = TimeSeriesClient(cid,model, x_train, y_train, x_test, y_test, x_val, y_val)
    
    fl.client.start_numpy_client("localhost:8080", client=client)

def run_simulation(train_dataframe, test_dataframe):
    
    processes = []
    server_process = Process(target=start_server)
    server_process.start()

    processes.append(server_process)

    time.sleep(2)
    
    partition = 10
    
    train_list_user = particiona(train_dataframe,partition)
    
    for i in range(len(train_list_user)):
    
        train_dataframe_user = train_list_user[i]
        client_process = Process(target=create_client, args=(i, train_dataframe_user, test_dataframe))
        client_process.start()
        processes.append(client_process)
     
    for process in processes:
        process.join()

class customFedAvg(fl.server.strategy.FedAvg):
    def aggregate_evaluate(
        self,
        rnd,
        results,
        failures: List[BaseException]
    ) -> Optional[float]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        losses = [r.metrics["loss"] * r.num_examples for _, r in results]
        recall = [r.metrics["recall"]* r.num_examples for _, r in results]
        f1 = [r.metrics["f1_score"]* r.num_examples for _, r in results]
        precision = [r.metrics["precision"]* r.num_examples for _, r in results]
        fpr = [r.metrics["fpr"] * r.num_examples for _, r in results]
        frr = [r.metrics["frr"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]
        sum_examples = sum(examples)
        # Aggregate and print custom metric
        accuracy_aggregated = sum(accuracies) / sum_examples
        loss_aggregated = sum(losses) / sum_examples
        recall_aggregated = sum(recall) / sum_examples
        precision_aggregated = sum(precision) / sum_examples
        f1_aggregated = sum(f1) / sum_examples
        fpr_aggregated = sum(fpr) / sum_examples
        frr_aggregated = sum(frr) / sum_examples
        print(f"Round {rnd} accuracy aggregated from client results: {accuracy_aggregated}")
        print(f"Round {rnd} recall aggregated from client results: {recall_aggregated}")
        print(f"Round {rnd} precision aggregated from client results: {precision_aggregated}")
        print(f"Round {rnd} FPR aggregated from client results: {fpr_aggregated}")
        print(f"Round {rnd} FRR aggregated from client results: {frr_aggregated}")
        print(f"Round {rnd} F1 aggregated from client results: {f1_aggregated}")
        print(f"Round {rnd} loss aggregated from client results: {loss_aggregated}")
        with open(path_server_aggregation_eval, 'a') as file:
            file.write(f"{loss_aggregated},{accuracy_aggregated},{recall_aggregated},{precision_aggregated},{fpr_aggregated},{frr_aggregated},{f1_aggregated},fedavg,{args.nn}")
            file.write('\n')
        # Call aggregate_evaluate from base class (FedAvg)
        return super().aggregate_evaluate(rnd, results, failures)
     
    def aggregate_fit(
        self,
        rnd,
        results,
        failures):
            
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        losses = [r.metrics["loss"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]
        # Aggregate and print custom metric
        accuracy_aggregated = sum(accuracies) / sum(examples)
        loss_aggregated = sum(losses) / sum(examples)
        print(f"Round {rnd} accuracy aggregated fitted from client results: {accuracy_aggregated}")
        print(f"Round {rnd} loss aggregated fitted from client results: {loss_aggregated}")
        with open(path_server_aggregation_fit, 'a') as file:
            file.write(f"{loss_aggregated},{accuracy_aggregated},fedavg,{args.nn}")
            file.write('\n')
        return super().aggregate_fit(rnd, results, failures)

class CustomFedAdagrad(fl.server.strategy.FedAdagrad):
   def aggregate_fit(
        self,
        rnd,
        results,
        failures):
            
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        losses = [r.metrics["loss"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]
        # Aggregate and print custom metric
        accuracy_aggregated = sum(accuracies) / sum(examples)
        loss_aggregated = sum(losses) / sum(examples)
        print(f"Round {rnd} accuracy aggregated fitted from client results: {accuracy_aggregated}")
        print(f"Round {rnd} loss aggregated fitted from client results: {loss_aggregated}")
        with open(path_server_aggregation_fit, 'a') as file:
            file.write(f"{loss_aggregated},{accuracy_aggregated},fedadagrad,{args.nn}")
            file.write('\n')
        return super().aggregate_fit(rnd, results, failures)
   def aggregate_evaluate(
        self,
        rnd,
        results,
        failures: List[BaseException]
    ) -> Optional[float]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        losses = [r.metrics["loss"] * r.num_examples for _, r in results]
        recall = [r.metrics["recall"] * r.num_examples for _, r in results]
        f1 = [r.metrics["f1_score"]* r.num_examples for _, r in results]
        precision = [r.metrics["precision"] * r.num_examples for _, r in results]
        fpr = [r.metrics["fpr"] * r.num_examples for _, r in results]
        frr = [r.metrics["frr"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]
        sum_examples = sum(examples)
        # Aggregate and print custom metric
        accuracy_aggregated = sum(accuracies) / sum_examples
        loss_aggregated = sum(losses) / sum_examples
        recall_aggregated = sum(recall) / sum_examples
        precision_aggregated = sum(precision) / sum_examples
        f1_aggregated = sum(f1) / sum_examples
        fpr_aggregated = sum(fpr) / sum_examples
        frr_aggregated = sum(frr) / sum_examples
        print(f"Round {rnd} accuracy aggregated from client results: {accuracy_aggregated}")
        print(f"Round {rnd} recall aggregated from client results: {recall_aggregated}")
        print(f"Round {rnd} precision aggregated from client results: {precision_aggregated}")
        print(f"Round {rnd} FPR aggregated from client results: {fpr_aggregated}")
        print(f"Round {rnd} FRR aggregated from client results: {frr_aggregated}")
        print(f"Round {rnd} F1 aggregated from client results: {f1_aggregated}")
        print(f"Round {rnd} loss aggregated from client results: {loss_aggregated}")
        with open(path_server_aggregation_eval, 'a') as file:
            file.write(f"{loss_aggregated},{accuracy_aggregated},{recall_aggregated},{precision_aggregated},{fpr_aggregated},{frr_aggregated},{f1_aggregated},fedadagrad,{args.nn}")
            file.write('\n')
        # Call aggregate_evaluate from base class (fedadagrad)
        return super().aggregate_evaluate(rnd, results, failures)

class customFedYogi(fl.server.strategy.FedYogi):
   def aggregate_fit(
        self,
        rnd,
        results,
        failures):
            
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        losses = [r.metrics["loss"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]
        # Aggregate and print custom metric
        accuracy_aggregated = sum(accuracies) / sum(examples)
        loss_aggregated = sum(losses) / sum(examples)
        print(f"Round {rnd} accuracy aggregated fitted from client results: {accuracy_aggregated}")
        print(f"Round {rnd} loss aggregated fitted from client results: {loss_aggregated}")
        #with open(path_server_aggregation_fit, 'a') as file:
        #    file.write(f"{loss_aggregated},{accuracy_aggregated},fedyogi,{args.nn}")
        #    file.write('\n')
        return super().aggregate_fit(rnd, results, failures)
   def aggregate_evaluate(
        self,
        rnd,
        results,
        failures: List[BaseException]
    ) -> Optional[float]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        losses = [r.metrics["loss"] * r.num_examples for _, r in results]
        recall = [r.metrics["recall"] * r.num_examples for _, r in results]
        f1 = [r.metrics["f1_score"] * r.num_examples for _, r in results]
        precision = [r.metrics["precision"] * r.num_examples for _, r in results]
        fpr = [r.metrics["fpr"] * r.num_examples for _, r in results]
        frr = [r.metrics["frr"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]
        sum_examples = sum(examples)
        # Aggregate and print custom metric
        accuracy_aggregated = sum(accuracies) / sum_examples
        loss_aggregated = sum(losses) / sum_examples
        recall_aggregated = sum(recall) / sum_examples
        precision_aggregated = sum(precision) / sum_examples
        f1_aggregated = sum(f1) / sum_examples
        fpr_aggregated = sum(fpr) / sum_examples
        frr_aggregated = sum(frr) / sum_examples
        print(f"Round {rnd} accuracy aggregated from client results: {accuracy_aggregated}")
        print(f"Round {rnd} recall aggregated from client results: {recall_aggregated}")
        print(f"Round {rnd} precision aggregated from client results: {precision_aggregated}")
        print(f"Round {rnd} FPR aggregated from client results: {fpr_aggregated}")
        print(f"Round {rnd} FRR aggregated from client results: {frr_aggregated}")
        print(f"Round {rnd} F1 aggregated from client results: {f1_aggregated}")
        print(f"Round {rnd} loss aggregated from client results: {loss_aggregated}")
        # loss,accuracy,recall,precision,fpr,frr,f1,strategy
        #with open(path_server_aggregation_eval, 'a') as file:
        #    file.write(f"{loss_aggregated},{accuracy_aggregated},{recall_aggregated},{precision_aggregated},{fpr_aggregated},{frr_aggregated},{f1_aggregated},fedyogi,{args.nn}")
        #    file.write('\n')
        # Call aggregate_evaluate from base class (fedadam)
        return super().aggregate_evaluate(rnd, results, failures)

def format_rows(X):
    save_rows = []
    for index, row in X.iterrows():
        new_row = []
        for i in range(len(row)):
            ts = pd.Series(map(float, row[i].split(',')))
            new_row.append(ts)
        save_rows.append(new_row)
    X_new = pd.DataFrame(save_rows)
    return X_new
    
def main():
    
    global groups

    # df.iloc[0:90, 'sample']

    test_dataframe = pd.read_csv(f'test_dataframe_{args.sensor}.csv', sep=':')
    train_dataframe = pd.read_csv(f'train_dataframe_{args.sensor}.csv', sep=':')
    
    groups = train_dataframe['label'].unique()

    onehot_encoder = OneHotEncoder(sparse=False,categories="auto")
    
    labels_train = train_dataframe['label'].values
    labels_train = labels_train.reshape(len(labels_train), 1)
    # generate coded labels for each class 
    encoded_labels_train = onehot_encoder.fit_transform(labels_train)
    train_dataframe['encoded_labels'] = np.nan
    train_dataframe['encoded_labels'] = train_dataframe['encoded_labels'].astype(object)
    # appends the coded label as an array in the dataframe.
    # this is more convenient
    for i in range(len(encoded_labels_train)):
        train_dataframe.at[i, 'encoded_labels'] = np.array(list((map(int, encoded_labels_train[i]))))
         
    labels_test = test_dataframe['label'].values
    labels_test = labels_test.reshape(len(labels_test), 1)
    encoded_labels_test = onehot_encoder.transform(labels_test)
    test_dataframe['encoded_labels'] = np.nan
    test_dataframe['encoded_labels'] = test_dataframe['encoded_labels'].astype(object)
    for i in range(len(encoded_labels_test)):
        test_dataframe.at[i, 'encoded_labels'] = np.array(list((map(int, encoded_labels_test[i]))))

    for i in range(0,10):
        run_simulation(train_dataframe, test_dataframe)

if __name__ == "__main__":
    main()
