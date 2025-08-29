from pathlib import Path
import os
import argparse
from subprocess import Popen,run,PIPE
import time



if __name__ == '__main__':
    """
    Script designed to apply the tombo resquiggle command to a whole dataset,
    it works on wick-like structure folders, adapting the command with the correct
    reference file for each species, either per-read or complete genome. 
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, help='dataset, structured as wick')
    parser.add_argument("--processes", type=int, help='number of workers at each time')
    args = parser.parse_args()
    """
    dataset_dir
        specie1
            fast5
            reference.fasta
        specie2 
        ...
    """
    process_handles=list()
    #cmds=list()
    for i,specie_dir in enumerate(os.listdir(args.dataset_dir)):
        if specie_dir not in ["tmp", "genomes"]:
            specie_name=specie_dir
            specie_dir=os.path.join(args.dataset_dir,specie_dir)

            ref_path=os.path.join(specie_dir,"read_references.fasta")

            if not os.path.exists(ref_path):
                ref_path=os.path.join(args.dataset_dir,'genomes/'+specie_name+"_reference.fna")

            cmd_str="tombo resquiggle "+os.path.join(specie_dir,"fast5")+" "\
            +" --processes 16 --dna --num-most-common-errors 5 --ignore-read-locks --overwrite"

            print("\n\ncurrent command:\n\n",i," ",cmd_str)
            #cmds.append(cmd_str)
            
            #run([cmd_str])
            os.system(cmd_str)
            """
            process_handles.append(Popen([cmd_str], shell=True,stdin=None, stdout=PIPE, stderr=PIPE, close_fds=True))
            
            if (i+1)%args.processes==0:         
                for j,handle in enumerate(process_handles):
                    print("\ncurrent command: ",cmds[j],"\n")
                    stdout_data=handle.stdout.read()
                    std_err=handle.stderr.read()
                    print("output:\n",stdout_data,std_err)
                process_handles.clear()
                cmds.clear()
            """
            
         