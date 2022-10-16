# -*- coding: utf-8 -*-
"""
train_net_valloss.py

Created on Tue Sep 20 09:25:29 2022

@author: Lukas


This script is based on the train_net.py script in the Detectron2 repo. 
"""

# import packages
import logging
import os
import json
from collections import OrderedDict

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import COCOEvaluator, verify_results
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.data import DatasetMapper, build_detection_test_loader

import pandas as pd
from LossEvalHook import LossEvalHook

# set up training environment
class Trainer(DefaultTrainer):
    """
    A trainer with default training logic. Creates a SimpleTrainer using model,
    optimizer, dataloader defined by the given config. Creates an LR scheduler
    defined by the config. Loads the last checkpoint or cfg.MODEL.WEIGHTS, 
    if it exists, when resume_or_load is called.
    
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Evaluate object proposals, AP for instance detection/segmentation
        
        Parameters
        ----------
        
        cfg : tuple[str]
            tasks that can be evaluated under the given configuration
            
        dataset_name : str
            name of the dataset to be evaluated (in json format)
            
        output_folder : str
            optional, an output directory to dump all results predicted on the 
            dataset. 
            
        Returns
        -------
        COCOEvaluator : object
        
        """
        return COCOEvaluator(dataset_name, cfg, True, output_folder)
    
    
    def build_hooks(self):
        """
        calls the separate LossEvalHook.py script 
        
        """
        hooks = super().build_hooks()
        hooks.insert(
            -1, LossEvalHook(
                self.cfg.TEST.EVAL_PERIOD,
                self.model, 
                build_detection_test_loader(
                    self.cfg, self.cfg.DATASETS.TEST[0],
                    DatasetMapper(self.cfg,True)
                    )
                )
            )
        return hooks
    

    @classmethod
    def test_with_TTA(cls, cfg, model):
        """
        test-time augmentation (TTA) for detection data
        
        Parameters
        ----------
        
        cfg : tuple[str]
            tasks that can be evaluated under the given configuration
            
        model : torch.nn.Module
            the current model as defined in the config file
        
        Returns
        -------
        
        res : dict
        
        """
        logger = logging.getLogger("detectron2.trainer")
        
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res
    
    
    #evaluation part: test time augmentation
    @classmethod
    def eval_and_save(cls, cfg, model):
        """
        [...]
        
        Parameters
        ----------
        
        cfg : tuple[str]
            tasks that can be evaluated under the given configuration
            
        model : torch.nn.Module
            the current model as defined in the config file
        
        Returns
        -------

        res : csv
            evaluation output, saved in the output directory
        
        """
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        pd.DataFrame(res).to_csv(os.path.join(cfg.OUTPUT_DIR, 'eval.csv'))
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    with open(args.json_annotation_train, 'r') as fp:
        anno_file = json.load(fp)

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(anno_file["categories"])
    del anno_file
    cfg.DATASETS.TRAIN = (f"{args.dataset_name}-train",)
    cfg.DATASETS.TEST = (f"{args.dataset_name}-val",)

    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    """
    main function: 
        registers the datasets, 
        runs inference, or
        trains the model
    
    """
    cfg = setup(args)
    
    # Register Datasets 
    dataset_name = args.dataset_name
    register_coco_instances(f"{dataset_name}-train", {}, args.json_annotation_train, args.image_path_train)
    register_coco_instances(f"{dataset_name}-val", {}, args.json_annotation_val, args.image_path_val)

    # if we're only evaluating the model:
    if args.eval_only:
        
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
    
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model)) 
        
        if comm.is_main_process():
            verify_results(cfg, res)

        # Save the evaluation results
        pd.DataFrame(res).to_csv(f'{cfg.OUTPUT_DIR}/eval.csv')
        return res
    
    # if we're training the model:
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.eval_and_save(cfg, trainer.model))]
    )
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


# parse arguments from shell
if __name__ == "__main__":
    parser = default_argument_parser()

    # Extra Configurations for dataset names and paths
    parser.add_argument("--dataset_name",          default="", help="The Dataset Name")
    parser.add_argument("--json_annotation_train", default="", metavar="FILE", help="The path to the training set JSON annotation")
    parser.add_argument("--image_path_train",      default="", metavar="FILE", help="The path to the training set image folder")
    parser.add_argument("--json_annotation_val",   default="", metavar="FILE", help="The path to the validation set JSON annotation")
    parser.add_argument("--image_path_val",        default="", metavar="FILE", help="The path to the validation set image folder")
    args = parser.parse_args()
    print("Command Line Args:", args)
    
    launch(main, args.num_gpus, 
           num_machines=args.num_machines, machine_rank=args.machine_rank, 
           dist_url=args.dist_url, args=(args,),)