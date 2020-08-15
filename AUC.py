import cv2
import numpy as np
import argparse
import os
import re
import progressbar

parser = argparse.ArgumentParser(description='AUC')

"""Main arguments"""
parser.add_argument('--gt_path', type=str, help='path to gt')
parser.add_argument('--disp_path', type=str, help='path to disp')
parser.add_argument('--conf_path', type=str, help='path to confidence')
parser.add_argument('--conf_name', type=str, help='confidence measure name')

"""Optional Arguments"""
parser.add_argument('--dataset', type=str, default='filelist/drivingstereo.txt', help='path to gt')
parser.add_argument('--tau', type=float, default=3, help='number of intervals for ROC curve')
parser.add_argument('--intervals', type=int, default=20, help='number of intervals for ROC curve')
parser.add_argument('--logfile', dest='logfile', type=str, default=None, help='Logfile')
args = parser.parse_args()

AUCs = []
opts = []
badts = []

samples = open(args.dataset).readlines()
samples = [s.split('.')[0] for s in samples]
samples.sort()
bar = progressbar.ProgressBar(max_value=len(samples))
counter=0

if args.logfile is not None:
        log = open(args.logfile, "w")

for i in (samples):

        gt = cv2.imread(args.gt_path+'/%s.png'%i, -1).astype(np.float32) / 256.
        disp = cv2.imread(args.disp_path+'/%s.png'%i, -1).astype(np.float32) / 256.
        conf = cv2.imread(args.conf_path+'/%s/'%(i)+args.conf_name+'.png', -1).astype(np.float32) / (256.*256.-1)

        valid = gt>0

        gt = gt[valid]
        disp = disp[valid]
        conf = conf[valid]

        badt = (np.abs(disp - gt) > args.tau).mean()

        ROC = []
        opt = []

        gt_conf = -np.abs(disp - gt)
        quants = [100./args.intervals*t for t in range(1,args.intervals)]

        thresholds = [np.percentile(conf, q) for q in quants]
        subs = [conf >= t for t in thresholds]
        ROC_points = [(np.abs(disp - gt) > args.tau)[s].mean() for s in subs]
        ROC.append(badt)
        [ROC.append(r) for r in ROC_points]
        ROC.append(0)

        gt_thresholds = [np.percentile(gt_conf, q) for q in quants]
        gt_subs = [gt_conf >= t for t in gt_thresholds]
        OPT_points = [(np.abs(disp - gt) > args.tau)[s].mean() for s in gt_subs]
        opt.append(badt)
        [opt.append(r) for r in OPT_points]
        opt.append(0)

        AUC = np.trapz(ROC, dx=1./args.intervals)
        AUCs.append(AUC)
        badts.append(badt)
        o = np.trapz(opt, dx=1./args.intervals)
        opts.append( o )

        if args.logfile is not None:
                log.write("%2.3f,%2.3f\n"%(AUC,o))

        counter +=1
        bar.update(counter)

avg_AUC = np.mean(np.array(AUCs))
avg_badt = np.array(badts).mean()

opt_AUC = np.array(opts).mean()
print('Measure: %s '%(args.conf_name)+'\t& Avg. bad%d: %2.3f%% \t& Opt. AUC: %.3f \t& Avg. AUC: %.3f \\\\'%(args.tau, avg_badt*100., opt_AUC, avg_AUC))

if args.logfile is not None:
        log.close()

