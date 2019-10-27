#! /usr/bin/python3

import ddff.dataproviders.datareaders.FocalStackDDFFH5Reader as FocalStackDDFFH5Reader
import ddff.metricseval.DDFFEval as DDFFEval
import pdb
import h5py

def main():
    #Set parameters
    image_size = (383,552)
    filename_testset = "../../ddff-dataset-trainval.h5"
    checkpoint_file = "checkpoints/ddff_cc3_checkpoint_1.pt"

    #Create validation reader
    tmp_datareader = FocalStackDDFFH5Reader.FocalStackDDFFH5Reader(filename_testset, transform=None, stack_key="stack_val", disp_key="disp_val")

    #Create PSPDDFF evaluator
    evaluator = DDFFEval.DDFFEval(checkpoint_file, focal_stack_size=tmp_datareader.get_stack_size())
    #Evaluate
    metrics = evaluator.evaluate(filename_testset, image_size=image_size)
    print(metrics)



if __name__ == "__main__":
    main()