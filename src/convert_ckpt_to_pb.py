import tensorflow.compat.v1 as tf
import sys

tf.disable_v2_behavior()
  
input_checkpoint = '../models/tmpmodel-'+sys.argv[1]+'.ckpt' # Your .meta file
output_node_name = ['dense_3/BiasAdd']
output_graph = '../models/model.pb'

def freeze_graph(input_checkpoint, output_graph):
	'''
	:param input_checkpoint:
	:param output_graph: PB model save path
	:return:
	''' 
	# Specify the output node name, the node name must be the node that exists in the original model
	
	saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)

	with tf.Session() as sess:
		saver.restore(sess, input_checkpoint)  # Restore the graph and get the data
		graph_def = tf.get_default_graph().as_graph_def()
		for n in graph_def.node:
		  print(n.name) 
		output_graph_def = tf.graph_util.convert_variables_to_constants(  # Model persistence, fixed variable values
				sess=sess,
				input_graph_def=graph_def,  # is equal to: sess.graph_def
				output_node_names=output_node_name)  # If there are multiple output nodes, use a comma Separate

		with tf.gfile.GFile(output_graph, "wb") as f:  # Save the model
			f.write(output_graph_def.SerializeToString())  # Serialize output     
    
freeze_graph(input_checkpoint,output_graph)           
