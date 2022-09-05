# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 10:10:55 2021

@author: Marius
"""


import torch.nn as nn
import torch
from datafactory import DataFactory
from torch.utils.data import DataLoader
from numpy import pi
import numpy as np
import time
from optimizers import SPSA, CMA
import datasets, models
import pickle
from sklearn.cross_validation import KFold

def train(args):
    start = time.time()
        
    save_name = (
        f"{args.tag}-{args.dataset}-{args.optimizer}-{args.learning_rate}-{args.model}-{args.n_blocks}-{args.n_qubits}-"
        + time.strftime("%m-%d--%H-%M")
    )
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    #Set loss function
    if args.loss == "BCE":
        criterion = nn.BCELoss()
    elif args.loss == "MSE":
        criterion = nn.MSELoss()
    elif args.loss == "CE":
        criterion = nn.CrossEntropyLoss()
    #Load data
    train_data = DataLoader(DataFactory(args.dataset+"-train"),
                            batch_size=args.batch_size, shuffle=True)
    test_data_ = DataFactory(args.dataset+"-test")
    test_data = DataLoader(test_data_,
                            batch_size=args.val_batch_size, shuffle=True)
    
    #Create model
    if args.model[:3] == "PQC":
        model = models.model_set[args.model](
            n_blocks=args.n_blocks,
            n_qubits=args.n_qubits,
            weights_spread=args.initial_weights_spread,
            grant_init=args.initialize_to_identity,
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
    
    losses = np.zeros((args.epochs))
    val_losses = np.zeros((args.epochs//10 + 1))
    accs = np.zeros((args.epochs//10 + 1))
    c = 0
    for epoch in range(args.epochs):
        x, y = next(iter(train_data))
        y = y.float()
        loss = None
        
        def loss_closure():
            nonlocal loss
            
            optimizer.zero_grad()
            state, output = model(x)
            pred = output.reshape(*y.shape)
            try:
                loss = criterion(pred, y)
            except RuntimeError:
                with open("fails/"+save_name+".fail", "a") as f:
                    print(args, file=f)                
                loss = criterion(pred, y)
            #Backpropagation
            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            
            return loss

        optimizer.step(loss_closure)
        
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         # print(name, param.data)
        #         # print(param.data[0,0,0,0,0].item())
        #         first_weight.append(param.data[0,0,0,0,0].item())
        
        losses[epoch] = loss.item()
        
        
        if epoch % 10 == 0:
            with torch.no_grad():
                x_val, y_val = next(iter(test_data))
                state, output = model(x_val)
                pred = output.reshape(*y_val.shape)
                y_val = y_val.float()
                
                val_loss = criterion(pred, y_val)
                val_losses[c] = val_loss.item()
                
                accs[c] = (torch.round(pred)==y_val).sum().item()/args.val_batch_size
                
                c += 1
                
    n_final_samples = min(len(test_data_), 500)
    final_data = DataLoader(test_data_, batch_size=n_final_samples)
    x_val, y_val = next(iter(final_data))
    state, output = model(x_val)
    pred = output.reshape(*y_val.shape)
    y_val = y_val.float()
    val_loss = criterion(pred, y_val)
    val_losses[c] = val_loss.item()
    accs[c] = (torch.round(pred)==y_val).sum().item()/n_final_samples
                
    end = time.time()
    results = {"training_loss": losses, "validation_loss": val_losses,
               "validation_accuracy": accs, "model_state_dict": model.state_dict(),
               "args": args, "timer": end-start,
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
        default=20,
        help="number of validation samples to draw each 10 epochs",
    )
    parser.add_argument(
        "--tag", metavar="TAG", type=str, default="", help="tag for logs"
    )
    parser.add_argument(
        "--epochs", metavar="EP", type=int, default=250, help="number of learning epochs"
    )
    parser.add_argument( #Momentarily dummy arguement
        "--seed",
        metavar="SEED",
        type=int,
        default=7,
        help="random seed for parameter initialization",
    )
    parser.add_argument(
        "--kfolds", metavar="KF", type=int, default=10, help="number of k-folds"
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
        help=f"dataset; choose between {', '.join(datasets.all_datasets)}",
    )
    parser_train.add_argument(
        "--loss",
        metavar="L",
        type=str,
        default="BCE",
        help="loss function, choose from BCE, MSE or CE",
    )
    parser_train.add_argument(
        "--batch-size", metavar="B", type=int, default=10, help="batch size",
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
        "--learning-rate",
        metavar="LR",
        type=float,
        default="0.05",
        help="learning rate for optimizer",
    )
    parser_train.add_argument(
        "--initial-weights-spread",
        metavar="IWσ",
        type=float,
        default=pi/2,
        help="initial weights spread for the parameterized Pauli rotation gates",
    )
    parser_train.add_argument(
        "--initialize-to-identity",
        metavar="INIT",
        type=bool,
        default=False,
        help="initialize weights to act as blocks of identities",
    )

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
    else:
        args.func(args)