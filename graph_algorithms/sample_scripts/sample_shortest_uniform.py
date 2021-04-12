import os

folder = 'data' 
sampling = 'uniform'
node_dim = 1
max_hop = 3
max_weight = 5
max_weight_tests = [5, 10]
train_graphs = ['path', 'cycle', 'ladder', '4regular', 'complete', 'tree', 'expander', 'general']
test_graph = 'general'
train_min_n = 20
train_max_n = 40
test_min_n = 50
test_max_n = 70
train_color = 5
test_color = 5

file_id = 0
for train_graph in train_graphs:
	for max_weight_test in max_weight_tests:
		data_name = f"shortestpath_uniform_{train_graph}_Ndim{node_dim}_maxhop{max_hop}_Train_V{train_min_n}_{train_max_n}_C{train_color}_E{max_weight}_Test_V{test_min_n}_{test_max_n}_C{test_color}_E{max_weight_test}"
		os.system('python shortest_generation.py --folder=%s --max_weight=%s --max_weight_test=%s --max_hop=%s --node_dim=%s --sampling=%s --train_graph=%s --test_graph=%s --train_min_n=%s --train_max_n=%s --test_min_n=%s --test_max_n=%s --train_color=%s --test_color=%s --data=%s'%(folder,max_weight,max_weight_test,max_hop,node_dim,sampling, train_graph, test_graph, train_min_n, train_max_n, test_min_n, test_max_n, train_color, test_color, data_name))
		file_id +=1

print("%d data files are successfully generated." %file_id)
