__author__ = 'slhuang'
import os
import subprocess

def profiling(dir_path, endwith, result_dir):
    files=sorted(os.listdir(dir_path))
    print (files)
    artifacts = ''
    for f in files:
        if f.endswith(endwith): #f.startswith('repo'): #
            artifacts += dir_path + f + ','
    print ("----------profiling ----------")
    proc = subprocess.Popen(['./src/profiling/profiling', '-dir', artifacts, '-result', result_dir], stdout=subprocess.PIPE)
    output = proc.communicate()[0]
    print('Got stdout:', str(output).replace('\\r', '\n').replace('\\n', '\n'))


def pre_clustering(clustering_strategy, result_dir):
    print ("---------pre_clustering----------")
    proc = subprocess.Popen(['./src/pre_clustering/pre_clustering', clustering_strategy, '-result', result_dir], stdout=subprocess.PIPE)
    output = proc.communicate()[0]
    print('Got stdout:', str(output).replace('\\r', '\n').replace('\\n', '\n'))

def clear_files(dir='.result/'):
    cmd = 'rm ' + dir + '*.csv'
    os.system(cmd)

def preserving_ops(dir_path, endwith, result_dir):
    files=sorted(os.listdir(dir_path))
    print (files)
    artifacts = ''
    for f in files:
        if f.endswith(endwith): #f.startswith('repo'): #
            artifacts += dir_path + f + ','
    print ("----------preserving-ops----------")
    proc = subprocess.Popen(['./src/preserving_ops/preserving_ops', '-dir', artifacts, '-result', result_dir], stdout=subprocess.PIPE)
    output = proc.communicate()[0]
    print('Got stdout:', str(output).replace('\\r', '\n').replace('\\n', '\n'))

def run():
    #convert_windows_to_linux('/home/slhuang/Public/2018-fall/lineage_inference/case_study/retail/')
    ### retail
    #profiling('/home/slhuang/Public/2018-fall/lineage_inference/case_study/retail/', '_linux.csv') #nb_123977/artifacts/')
    #pre_clustering('-exact_schema')
    #preserving_ops('/home/slhuang/Public/2018-fall/lineage_inference/case_study/retail/') #nb_123977/artifacts/') # /repo_user/')
    
    ### nb_123977
    #profiling('/home/slhuang/Public/2018-fall/lineage_inference/case_study/nb_123977/artifacts/', '.csv') #retail/') #nb_123977/artifacts/')
    #pre_clustering('-exact_schema')
    #preserving_ops('/home/slhuang/Public/2018-fall/lineage_inference/case_study/nb_123977/artifacts/', '.csv') #nb_123977/artifacts/') # /repo_user/')
    result_dir = "./test/"
    clear_files(result_dir)
    profiling('./dataset/', '.csv', result_dir) #retail/') #nb_123977/artifacts/')
    pre_clustering('-exact_schema', result_dir)  #('-no_pre_cluster')#
    preserving_ops('./dataset/', '.csv', result_dir) #nb_123977/artifacts/') # /repo_user/')
    
if __name__ == '__main__':
    run()