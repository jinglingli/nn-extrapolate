import os

folder = 'data'
node_dim = 1
sampling = 'identical'
train_graphs = ['path', 'cycle', 'ladder', '4regular', 'complete', 'tree', 'expander', 'general']
test_graph = 'general'
train_min_n = 20
train_max_n = 30
test_min_ns = [50]
train_color = 1
test_colors = [1]

file_id = 0
for train_graph in train_graphs:
	for test_min_n in test_min_ns:
		if test_min_n == 20:
			test_max_n = 30
		else:
			test_max_n = 100
		for test_color in test_colors:
			data_name = f"maxdeg_identical_{train_graph}_Ndim{node_dim}_Train_V{train_min_n}_{train_max_n}_C{train_color}_Test_V{test_min_n}_{test_max_n}_C{test_color}"
			os.system('python maxdeg_generation.py --folder=%s --node_dim=%s --sampling=%s --train_graph=%s --test_graph=%s --train_min_n=%s --train_max_n=%s --test_min_n=%s --test_max_n=%s --train_color=%s --test_color=%s --data=%s'%(folder,node_dim,sampling, train_graph, test_graph, train_min_n, train_max_n, test_min_n, test_max_n, train_color, test_color, data_name))
			file_id +=1

print("%d data files are successfully generated." %file_id)
