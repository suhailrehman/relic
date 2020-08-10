
#BASE_DIR='/home/suhail/ok/'
#BASE_DIR='/home/suhail/Projects/sample_workflows/million_notebooks/selected/'
BASE_DIR='/home/suhail/Projects/relic/primitives/python/generator/dataset/'

nb_name='20200720-111903'

#rm -rf $BASE_DIR$nb_name'/inferred/'
#rm -rf $BASE_DIR$nb_name'/inferred/'

#python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering=PC2
#python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='No Precluster' --metric=col --swap=True --recompute=True



#Rerun:
#20200517-200627 cell+detectors
#20200517-200536 detectors
#20200517-201531 cell+detectors


#python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='No Precluster' --metric=cell --recompute=True
#python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='No Precluster' --metric=cell --recompute=True --group=True
#python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='No Precluster' --metric=cell --recompute=True --join=True
#python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='PC2' --metric=cell --recompute=True --join=True --group=True --transform=True --pivot=True


#python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='PC2' --metric='pc2cellcoltimestamp' --recompute=True --group=True --pivot=True
python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='PC2' --metric='pc2cellgt' --recompute=True --join=True --group=True --pivot=True --colt=0.8

#python agglomerative.py --basedir=$BASE_DIR --nbname=20200517-200536 --clustering='No Precluster' --metric=cell --recompute=True --join=True --group=True --transform=True --pivot=True
#python agglomerative.py --basedir=$BASE_DIR --nbname=20200517-200536 --clustering='PC2' --metric=pc2cellcol --recompute=True --join=True --group=True --transform=True --pivot=True
#python agglomerative.py --basedir=$BASE_DIR --nbname=20200517-201531 --clustering='No Precluster' --metric=cell --recompute=True --join=True --group=True --transform=True --pivot=True

#20200517-200732 to be done