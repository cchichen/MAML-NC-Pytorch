import  torch, os
import  numpy as np
from    MiniImagenet import MiniImagenet
import  scipy.stats
from    torch.utils.data import DataLoader
from    torch.optim import lr_scheduler
import  random, sys, pickle
import  argparse
import matplotlib.pyplot as plt

from meta import Meta


def mean_confidence_interval(accs, confidence=0.95):
    n = accs.shape[0]
    m, se = np.mean(accs), scipy.stats.sem(accs)
    h = se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)
    return m, h


def main():

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print(args)

    config = [
        ('conv2d', [32, 3, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 1, 0]),
        ('flatten', []),
        ('linear', [args.n_way, 32 * 5 * 5])
    ]

    device = torch.device('cuda')
    maml = Meta(args, config).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    # batchsz here means total episode number
    mini = MiniImagenet('miniimagenet/', mode='train', n_way=args.n_way, k_shot=args.k_spt,
                        k_query=args.k_qry,
                        batchsz=10000, resize=args.imgsz)
    mini_test = MiniImagenet('miniimagenet/', mode='test', n_way=args.n_way, k_shot=args.k_spt,
                             k_query=args.k_qry,
                             batchsz=100, resize=args.imgsz)

    # wth_traces_meta = []
    # btw_traces_meta = []
    # relative_wth_trace = []
    # relative_btw_trace = []

    # # graph for finetuning
    # wth_traces_finetune = []
    # btw_traces_finetune = []
    # relative_wth_trace_finetune = []
    # relative_btw_trace_finetune = []

    for epoch in range(args.epoch//10000):
        wth_traces_meta = []
        btw_traces_meta = []
        relative_wth_trace = []
        relative_btw_trace = []

        # graph for finetuning
        wth_traces_finetune = []
        btw_traces_finetune = []
        relative_wth_trace_finetune = []
        relative_btw_trace_finetune = []
        # fetch meta_batchsz num of episode each time
        db = DataLoader(mini, args.task_num, shuffle=True, num_workers=1, pin_memory=True)

        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):

            x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)

            accs, wth_meta, btw_meta, wth_fine, btw_fine = maml(x_spt, y_spt, x_qry, y_qry)

            

            if step % 100 == 0:
                print('step:', step, '\ttraining acc:', accs)
                print('Meta wth:',wth_meta,'\tbtw',btw_meta,"\tFinetuned, wth trace:", wth_fine, "\tbtw trace:",btw_fine)
                # Store the traces for Meta model
                total_trace = wth_meta + btw_meta

                # Avoid division by zero
                if total_trace != 0:
                    relative_trace_wth = wth_meta / total_trace
                    relative_trace_btw = btw_meta / total_trace
                else:
                    relative_trace_btw = 0
                    relative_trace_wth = 0


                wth_traces_meta.append(wth_meta)
                btw_traces_meta.append(btw_meta)
                relative_wth_trace.append(relative_trace_wth)
                relative_btw_trace.append(relative_trace_btw)
                
                # Store the traces for finetuned model
                total_trace_fine = wth_fine + btw_fine

                # Avoid division by zero
                if total_trace_fine != 0:
                    relative_trace_wth_fine = wth_fine / total_trace_fine
                    relative_trace_btw_fine = btw_fine / total_trace_fine
                else:
                    relative_trace_wth_fine = 0
                    relative_trace_btw_fine = 0

                # Store the traces
                wth_traces_finetune.append(wth_fine)
                btw_traces_finetune.append(btw_fine)
                relative_wth_trace_finetune.append(relative_trace_wth_fine)
                relative_btw_trace_finetune.append(relative_trace_btw_fine)

            if step % 500 == 0:  # evaluation
                db_test = DataLoader(mini_test, 1, shuffle=True, num_workers=1, pin_memory=True)
                accs_all_test = []


                relative_wth_fine_all = []
                relative_btw_fine_all = []

                for x_spt, y_spt, x_qry, y_qry in db_test:
                    x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                                 x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)

                    accs, wth_trace_fine, btw_trace_fine = maml.finetunning(x_spt, y_spt, x_qry, y_qry)

                    accs_all_test.append(accs)
                    # print('fine_wth:',wth_trace_fine,'\tfine_btw',btw_trace_fine,"\tfine_relative wth trace:", relative_trace_wth_fine, "\fine_relative btw trace:",relative_trace_btw_fine)

                # [b, update_step+1]
                accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
                print('Test acc:', accs)
                # wth_all = np.array(relative_wth_fine_all).mean(axis=0).astype(np.float16)
                # print('Finetuned Relative within:',wth_all )
                # btw_all = np.array(relative_btw_fine_all).mean(axis=0).astype(np.float16)
                # print('Finetuned Relative between:',btw_all )


            # steps = range(len(wth_trace_over_steps))
            # print('steps:', steps)
            # print('relative_wth_trace_over_steps:', relative_wth_trace_over_steps)
            # print('wth_trace_over_steps', wth_trace_over_steps)



            # # Plotting each trace
            # plt.plot(steps, wth_trace_over_steps, label='Within Trace')
            # plt.plot(steps, btw_trace_over_steps, label='Between Trace')
            # plt.plot(steps, relative_wth_trace_over_steps, label='Relative Within Trace')

            # # Adding labels and title
            # plt.xlabel('Steps')
            # plt.ylabel('Trace Value')
            # plt.title('Trace Values Over Steps')

            # # Add a legend
            # plt.legend()

            # # Show the plot
            # plt.show()
            # Plotting after training
            
        # print("within total traces", wth_traces_meta)
        # print("betwn total traces", btw_traces_meta)
        plt.figure(figsize=(12, 12))

        steps = range(len(wth_traces_meta))
        plt.subplot(421)
        plt.plot(wth_traces_meta, label='Meta Within-Class Covariance Trace')
        plt.plot(btw_traces_meta, label='Meta Between-Class Covariance Trace')
        plt.xlabel('Steps(unit = 100)')
        plt.ylabel('Trace')
        plt.title('Meta model Trace of Covariance Matrices')
        plt.legend()

        plt.subplot(422)
        plt.plot(relative_wth_trace, label='Relative Within-Class Trace')
        plt.plot(relative_btw_trace, label='Relative Between-Class Trace')
        plt.xlabel('Steps(unit = 100)')
        plt.ylabel('Relative Trace')
        plt.title('Meta Model Relative Traces')
        plt.legend()


        # subplot 3: finetune traces
        plt.subplot(423)
        plt.plot(wth_traces_finetune, label='Finetuned Within-Class Covariance Trace')
        plt.plot(btw_traces_finetune, label='Finetuned Between-Class Covariance Trace')
        plt.xlabel('Steps(unit = 100)')
        plt.ylabel('Trace')
        plt.title('Fine Tuned - Trace of Covariance Matrices over Steps')
        plt.legend()

        # Subplot 4: finetune relative traces
        plt.subplot(424)
        plt.plot(relative_wth_trace_finetune, label='Finetuned Relative Within-Class Trace')
        plt.plot(relative_btw_trace_finetune, label='Finetuned Relative Between-Class Trace')
        plt.xlabel('Steps(unit = 100)')
        plt.ylabel('Relative Trace')
        plt.title('Finetuned - Relative Traces over Steps')
        plt.legend()




        plt.tight_layout()
        plt.show()
        

  

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=60000)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=5)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=84)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=4)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)

    args = argparser.parse_args()

    main()
