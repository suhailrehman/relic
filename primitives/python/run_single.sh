
#BASE_DIR='/home/suhail/ok/'
BASE_DIR='/home/suhail/Projects/sample_workflows/million_notebooks/selected/'

nb_name='agri-mex'

#python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering=PC2
#python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='No Precluster' --metric=cell --recompute=True

python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='No Precluster' --metric=cell --recompute=True --join=True
#python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='No Precluster' --metric=cell --recompute=True --group=True

