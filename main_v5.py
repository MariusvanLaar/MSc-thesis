# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 10:10:55 2021

@author: Marius
"""


import torch.nn as nn
import torch
from datasets.datafactory import DataFactory, FoldFactory
from torch.utils.data import DataLoader
from numpy import pi
import numpy as np
import time
from optimizers import SPSA, CMA
import datasets, models
import pickle
from sklearn.model_selection import KFold
import glob, os


def train(args):
    
    torch.manual_seed(args.seed+123456789) #To create a large number with a good balance 
    np.random.seed(args.seed+123456789) # of 0 and 1 bits
    
    n_features = args.n_blocks*args.n_qubits
    dataclass = datasets.all_datasets[args.dataset](n_features)
    assert "loss" in dataclass.data_info, "Dataclass has no assigned loss function"
    kf = KFold(args.kfolds, shuffle=True, random_state=args.seed)
    for fold, (train_idx, test_idx) in enumerate(kf.split(dataclass.data)):
        save_name = f"{args.tag}-{fold}-{args.dataset}-{args.optimizer}-{args.learning_rate}-" \
        f"{args.model}-{args.n_layers}-{args.n_blocks}-{args.n_qubits}-{args.seed}-*"
        if len(glob.glob("runs"+os.sep+save_name)) == 0: #Check if this iteration has been done before
            X_tr, Y_tr = dataclass[train_idx]
            X_te, Y_te = dataclass[test_idx]
            
            dataclass.fit(X_tr.copy())
            X_tr = dataclass.transform(X_tr)
            X_te = dataclass.transform(X_te)
                    
            train_set = FoldFactory(X_tr, Y_tr)
            test_set = FoldFactory(X_te, Y_te)
                    
            train_(train_set, test_set, fold, {**args, **dataclass.data_info})
        

def train_(train_set, test_set, fold_id, args):
    start = time.time()
        
    save_name = (
        f"{args.tag}-{fold_id}-{args.dataset}-{args.optimizer}-{args.learning_rate}-" 
        + f"{args.model}-{args.n_layers}-{args.n_blocks}-{args.n_qubits}-{args.seed}-"
        + time.strftime("%m-%d--%H-%M")
    )
    
    torch.manual_seed(args.seed+fold_id+123456789)
    np.random.seed(args.seed+fold_id+123456789)
    
    #Set loss function
    if args.loss == "BCE":
        criterion = nn.BCELoss()
    elif args.loss == "MSE":
        criterion = nn.MSELoss()
    elif args.loss == "CE":
        criterion = nn.CrossEntropyLoss()
    #Load data
    train_data = DataLoader(train_set,
                            batch_size=args.batch_size, shuffle=True)
    val_batch_size = min(len(test_set), args.val_batch_size)
    test_data = DataLoader(test_set,
                            batch_size=val_batch_size, shuffle=True)
    
    #Create model
    if args.model[:3] == "PQC":
        model = models.model_set[args.model](
            n_blocks=args.n_blocks,
            n_qubits=args.n_qubits,
            n_layers=args.n_layers,
            observable=args.observable,
            return_prob=args.return_prob,
            weights_spread=args.initial_weights_spread,
            )
    elif args.model[:3] == "ANN":
        model = models.model_set[args.model](
            input_dim=args.n_blocks*args.n_qubits,
            )
            
    #Create optimizer
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            )
    elif args.optimizer == "spsa":
        optimizer = SPSA(
            model.parameters(),
            lr=args.learning_rate,
            )
    elif args.optimizer == "lbfgs":
        optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr=args.learning_rate,
            history_size=25,
            )
    elif args.optimizer == "cma":
        optimizer = CMA(
            model.parameters(),
            lr=args.learning_rate,
            )
    
    losses = np.zeros((args.epochs+1))
    t_accs = np.zeros((args.epochs+1))
    gradient_1, gradient_2 = np.zeros((args.epochs)), np.zeros((args.epochs))
    val_losses = np.zeros((args.epochs//10 + 1))
    v_accs = np.zeros((args.epochs//10 + 1))
    c = 0
    for epoch in range(args.epochs):
        x, y = next(iter(train_data))
        y = y.float()
        loss = None
        training_acc = None
        
        def loss_closure():
            nonlocal loss
            nonlocal training_acc
            
            optimizer.zero_grad()
            output = model(x)
            if args.return_probs:
                output = model.return_probability(output)
            pred = output.reshape(*y.shape)
            training_acc = (torch.round(pred)==y).sum().item()/args.batch_size
           
            try:
                loss = criterion(pred, y)
            except RuntimeError:
                with open("fails/"+save_name+".fail", "a") as f:
                    print(args, file=f)                
                loss = criterion(pred, y)
            #Backpropagation
            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            
            return loss

        optimizer.step(loss_closure)
        
        gradient_1[epoch] = model.var[0][0].weights.grad.flatten()[-1].item() #Gradient first Yrot of final qubit in final block
        gradient_2[epoch] = model.fvar[1].weights.grad.flatten()[-1].item() #Gradient last Yrot of final qubit in final block
        losses[epoch] = loss.item()
        t_accs[epoch] = training_acc
        
        
        if epoch % 10 == 0:
            with torch.no_grad():
                x_val, y_val = next(iter(test_data))
                output = model(x_val)
                if args.return_probs:
                    output = model.return_probability(output)
                pred = output.reshape(*y_val.shape)
                y_val = y_val.float()
                
                val_loss = criterion(pred, y_val)
                val_losses[c] = val_loss.item()
                
                v_accs[c] = (torch.round(pred)==y_val).sum().item()/val_batch_size
                
                c += 1
                
    with torch.no_grad():
        n_final_t_samples = min(len(train_set), 500)
        final_t_data = DataLoader(train_set, batch_size=n_final_t_samples)                
        x, y = next(iter(final_t_data))
        output = model(x)
        if args.return_probs:
            output = model.return_probability(output)
        pred = output.reshape(*y.shape)
        y = y.float()
        loss = criterion(pred, y)
        losses[-1] = loss.item()
        t_accs[-1] = (torch.round(pred)==y).sum().item()/n_final_t_samples 
        
        
        n_final_samples = min(len(test_set), 500)
        final_data = DataLoader(test_set, batch_size=n_final_samples)
        x_val, y_val = next(iter(final_data))
        output = model(x_val)
        if args.return_probs:
            output = model.return_probability(output)
        pred = output.reshape(*y_val.shape)
        y_val = y_val.float()
        val_loss = criterion(pred, y_val)
        val_losses[c] = val_loss.item()
        v_accs[c] = (torch.round(pred)==y_val).sum().item()/n_final_samples
                
    end = time.time()
    results = {"training_loss": losses, "training_acc": t_accs,
               "validation_loss": val_losses,
               "validation_accuracy": v_accs,
               "args": args, "timer": end-start,
               "gradient1": gradient_1, "gradient2": gradient_2,
               }
    pickling = open("runs/"+save_name+".pkl", "wb")
    pickle.dump(results, pickling)
    pickling.close()
    

def command_train(args):
    train(args)



if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="QML Training Script", formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--val-batch-size",
        metavar="VALS",
        type=int,
        default=32,
        help="number of validation samples to draw each 10 epochs",
    )
    parser.add_argument(
        "--tag", metavar="TAG", type=str, default="", help="tag for logs"
    )
    parser.add_argument(
        "--epochs", metavar="EP", type=int, default=250, help="number of learning epochs"
    )
    parser.add_argument(
        "--seed",
        metavar="SEED",
        type=int,
        default=7,
        help="random seed for parameter initialization",
    )
    parser.add_argument(
        "--kfolds", metavar="KF", type=int, default=5, help="number of k-folds"
    )
    
    subparsers = parser.add_subparsers(help="available commands")

    parser_train = subparsers.add_parser(
        "train", formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_train.set_defaults(func=command_train)

    parser_train.add_argument(
        "--dataset",
        metavar="D",
        type=str,
        default="wdbc",
        help=f"dataset; choose between {', '.join(datasets.all_datasets.keys())}",
    )
    parser_train.add_argument(
        "--batch-size", metavar="B", type=int, default=32, help="batch size",
    )
    parser_train.add_argument(
        "--optimizer",
        metavar="OPT",
        type=str,
        default="adam",
        help="optimizer; one of adam, SPSA, LFBGS",
    )
    parser_train.add_argument(
        "--model",
        metavar="MODEL",
        type=str,
        default="PQC-1A",
        help=f"model; choose between {', '.join(models.model_set.keys())}",
    )
    parser_train.add_argument(
        "--n-blocks",
        metavar="NB",
        type=int,
        default=2,
        help="number of blocks in model",
    )
    parser_train.add_argument(
        "--n-qubits",
        metavar="NQ",
        type=int,
        default=5,
        help="number of qubits per block",
    )
    parser_train.add_argument(
        "--n-layers",
        metavar="NL",
        type=int,
        default=5,
        help="number of layers in certain models",
    )
    parser_train.add_argument(
        "--observable",
        metavar="OB",
        default="All",
        help="type of observable",
    )
    parser_train.add_argument(
        "--return-prob",
        metavar="P",
        default=False,
        help="Boolean of whether to map the model output to the range [0,1]",
    )
    parser_train.add_argument(
        "--learning-rate",
        metavar="LR",
        type=float,
        default="0.05",
        help="learning rate for optimizer",
    )
    parser_train.add_argument(
        "--initial-weights-spread",
        metavar="IWσ",
        type=list,
        default=[-pi/2, pi/2],
        help="initial weights spread for the parameterized Pauli rotation gates",
    )

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
    else:
        args.func(args)