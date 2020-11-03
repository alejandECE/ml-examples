#  Created by Luis Alejandro (alejand@umich.edu).
#  Copyright © Do not distribute or use without authorization from author

import argparse
import model

if __name__ == '__main__':
  # Creates argument parser
  parser = argparse.ArgumentParser()
  # Trains model
  model.train_and_evaluate()
