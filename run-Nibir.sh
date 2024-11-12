for dataset in 'ogbn-arxiv'
do
    python -m core.trainEnsemble dataset $dataset gnn.model.name SAGE >> ${dataset}_sage_test.out
done