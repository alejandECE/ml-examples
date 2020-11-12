#  Created by Luis Alejandro (alejand@umich.edu).
#  Copyright © Do not distribute or use without authorization from author

import argparse
import model

if __name__ == '__main__':
  # Creates argument parser
  parser = argparse.ArgumentParser()
  parser.add_argument('--state', help='US state to work with (Default: MI)', type=str)
  parser.add_argument('--version', help='Version of the datasets to load (Default: None)', type=int)
  parser.add_argument('--epochs', help='Number of epochs to train (Default: 100)', type=int)
  parser.add_argument('--model_type', help='Type of model use (Options: linear, dnn, rnn', type=str)
  args = parser.parse_args()
  model.train_and_evaluate(args)
