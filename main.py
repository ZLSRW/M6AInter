import os
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
from datetime import datetime
from models.handler import train, test
import argparse
from data_loader.SiteBinding_dataloader1 import *

parser = argparse.ArgumentParser()
parser.add_argument('--train', type=bool, default=True)
parser.add_argument('--dataset', type=str, default='Motifs')
parser.add_argument('--optimizer', type=str, default='RMSProp')
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--multi_layer', type=int, default=5)
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--batch_size', type=int)
parser.add_argument('--decay_rate', type=float, default=0.1) #0.5
parser.add_argument('--dropout_rate', type=float, default=0.8) #0.5
parser.add_argument('--leakyrelu_rate', type=int, default=0.5) #0.2
parser.add_argument('--seq_len', type=int, default=41) #0.2
parser.add_argument('--seq_len1', type=int, default=64) #0.2


parser.add_argument('--cluster_num', type=int, default=41)


args = parser.parse_args()
print(f'Training configs: {args}')

result_train_file = os.path.join('output', args.dataset, 'Human_Kidney')
result_test_file = os.path.join('output', args.dataset, 'test')
if not os.path.exists(result_train_file):
    os.makedirs(result_train_file)
if not os.path.exists(result_test_file):
    os.makedirs(result_test_file)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    if args.train: #训练加验证
        try:
            before_train = datetime.now().timestamp()
            i=0
            all_result=[]
            while i<5:
                print('fold '+str(i)+' ')
                print('-'*99)
                train_data = []
                valid_data = []
                ReadMyCsv(train_data,'./samples/Train'+str(i)+'.csv')

                ReadMyCsv(valid_data,'./samples/Test'+str(i)+'.csv')

                args.batch_size=len(valid_data)+1
                print(args.batch_size)
                print('Train begining!')
                forecast_feature,result=train(train_data, valid_data, args, result_train_file,i)
                all_result.append(result)

                i+=1
            after_train = datetime.now().timestamp()
            print(f'Training took {(after_train - before_train) / 60} minutes')

        except KeyboardInterrupt:
            print('-' * 99)
            print('Exiting from training early')
    # if args.evaluate:
    #     before_evaluation = datetime.now().timestamp()
    #     test(test_data, args, result_train_file, result_test_file)
    #     after_evaluation = datetime.now().timestamp()
    #     print(f'Evaluation took {(after_evaluation - before_evaluation) / 60} minutes')
    print('done')
    """
    设置不同的任务，包括结合位点预测（main task）、亲和力预测。甚至可以重新考虑数据集。
    """



