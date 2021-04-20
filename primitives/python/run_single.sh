
#BASE_DIR='/home/suhail/ok/'
#BASE_DIR='/home/suhail/Projects/sample_workflows/million_notebooks/selected/'
#BASE_DIR='/home/suhail/Projects/sample_workflows/million_notebooks/new_selection/'
#BASE_DIR='/home/suhail/Projects/relic/primitives/python/generator/dataset_flat_exact/'
#BASE_DIR="/home/suhail/Projects/relic/primitives/python/generator/dataset_flat_timing/"
BASE_DIR="/home/suhail/Projects/sample_workflows/relic_datasets_vldb_2021/mixed/"


#nb_name='20210126-171907'

nb_name='basenb'
nb_name='syn2'


#nb_name='nyc-property'
#nb_name='london-crime'

#rm -rf $BASE_DIR$nb_name'/inferred/'
#rm -rf $BASE_DIR$nb_name'/inferred/'

#python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='No Precluster' --metric=cell --recompute=True --cellt=-1.0

#python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='No Precluster' --metric=cell --recompute=True --join=True --group=True  --cellt=0.1 --pivot=True

#python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='PC2' --metric='pc2cell+containment' --recompute=True --group=True --join=True  --cellt=0.1 --pivot=True

#python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='PC2' --metric='pc2cellcolj1' --recompute=True --group=True --join=True  --colt=0.3 --cellt##=0.1 --pivot=True

#python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='No Precluster' --metric=colms --swap=True --recompute=True --colt=0

#python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='No Precluster' --metric=colmscon --swap=True --recompute=True --colt=0

#python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='No Precluster' --metric=colmscon --recompute=True --join=True --group=True --cellt=0.1 --pivot=True

#python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='PC2' --metric=pc2colmscon --recompute=True --group=True --join=True --cellt=0.1 --pivot=True

#python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='No Precluster' --metric=col --swap=True --recompute=True --colt=0


#python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='PC2' --metric='pc2cc_con' --recompute=True --group=True --join=True  --cellt=0.1 --pivot=True

#i=1.0
#j=1.0
#python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='PC2' --metric="pc2cell_${i}_${j}+containment" --recompute=True --cellt=$i --intercellt=$j --group=True --join=True  --pivot=True


#python agglomerative_full.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='PC2' --metric='pc2cell+containment' --group=True --join=True --cellt=0.1 --pivot=True --transform=True

python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='PC2' --metric='pc2cell+containment' --recompute=True  --cellt=0.1 --join=True --pivot=True --group=True

