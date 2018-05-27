from tensorflow.python.tools import inspect_checkpoint

# inspect_checkpoint.print_tensors_in_checkpoint_file('/tmp/models/inception_v3/model.ckpt-20', tensor_name='', all_tensors=True, all_tensor_names=True)

inspect_checkpoint.print_tensors_in_checkpoint_file('/tmp/models/inception_v3/model.ckpt-20', tensor_name='InceptionV3/Conv2d_2a_3x3/BatchNorm/beta/RMSProp_1', all_tensors=False)
