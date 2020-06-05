
#BASE_DIR='/home/suhail/ok/'
BASE_DIR='/home/suhail/Projects/sample_workflows/million_notebooks/selected/'
#BASE_DIR='/home/suhail/Projects/relic/primitives/python/generator/dataset/'

nb_name='prop64-new'

#python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering=PC2


#Rerun:
#20200517-200627 cell+detectors
#20200517-200536 detectors
#20200517-201531 cell+detectors


#python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='No Precluster' --metric=cell --recompute=True
#python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='No Precluster' --metric=cell --recompute=True --group=True
python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='No Precluster' --metric=cell --recompute=True --join=True
#python agglomerative.py --basedir=$BASE_DIR --nbname=$nb_name --clustering='No Precluster' --metric=cell --recompute=True --join=True --group=True --transform=True --pivot=True


#python agglomerative.py --basedir=$BASE_DIR --nbname=20200517-200627 --clustering='No Precluster' --metric=cell --recompute=True --join=True --group=True --transform=True --pivot=True
#python agglomerative.py --basedir=$BASE_DIR --nbname=20200517-200536 --clustering='No Precluster' --metric=cell --recompute=True --join=True --group=True --transform=True --pivot=True
#python agglomerative.py --basedir=$BASE_DIR --nbname=20200517-200536 --clustering='PC2' --metric=pc2cellcol --recompute=True --join=True --group=True --transform=True --pivot=True
#python agglomerative.py --basedir=$BASE_DIR --nbname=20200517-201531 --clustering='No Precluster' --metric=cell --recompute=True --join=True --group=True --transform=True --pivot=True

#20200517-200732 to be done