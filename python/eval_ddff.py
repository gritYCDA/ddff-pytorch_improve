#! /usr/bin/python3

import ddff.dataproviders.datareaders.FocalStackDDFFH5Reader as FocalStackDDFFH5Reader
import ddff.metricseval.DDFFEval as DDFFEval
import pdb
import h5py

if __name__ == "__main__":
    #Set parameters
    image_size = (224, 244)
    filename_testset = "../../ddff-dataset-trainval.h5"
    checkpoint_file = "checkpoints/ddff_cc3_checkpoint_3.pt"

    #Create validation reader
    tmp_datareader = FocalStackDDFFH5Reader.FocalStackDDFFH5Reader(filename_testset, transform=None, stack_key="stack_val", disp_key="disp_val")

    tmp_datareader.get_raw_input()

    #Create PSPDDFF evaluator
    evaluator = DDFFEval.DDFFEval(checkpoint_file, focal_stack_size=tmp_datareader.get_stack_size())
    #Evaluate
    metrics = evaluator.evaluate(filename_testset, image_size=image_size)
    print(metrics)
