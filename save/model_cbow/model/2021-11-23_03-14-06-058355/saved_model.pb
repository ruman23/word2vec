Ъ
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.6.02unknown8??
?
w2v_embedding_cbow/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??**
shared_namew2v_embedding_cbow/kernel
?
-w2v_embedding_cbow/kernel/Read/ReadVariableOpReadVariableOpw2v_embedding_cbow/kernel* 
_output_shapes
:
??*
dtype0
?
w2v_embedding_cbow/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*(
shared_namew2v_embedding_cbow/bias
?
+w2v_embedding_cbow/bias/Read/ReadVariableOpReadVariableOpw2v_embedding_cbow/bias*
_output_shapes	
:?*
dtype0
v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
??*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:?*
dtype0
l
RMSprop/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameRMSprop/iter
e
 RMSprop/iter/Read/ReadVariableOpReadVariableOpRMSprop/iter*
_output_shapes
: *
dtype0	
n
RMSprop/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/decay
g
!RMSprop/decay/Read/ReadVariableOpReadVariableOpRMSprop/decay*
_output_shapes
: *
dtype0
~
RMSprop/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameRMSprop/learning_rate
w
)RMSprop/learning_rate/Read/ReadVariableOpReadVariableOpRMSprop/learning_rate*
_output_shapes
: *
dtype0
t
RMSprop/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameRMSprop/momentum
m
$RMSprop/momentum/Read/ReadVariableOpReadVariableOpRMSprop/momentum*
_output_shapes
: *
dtype0
j
RMSprop/rhoVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/rho
c
RMSprop/rho/Read/ReadVariableOpReadVariableOpRMSprop/rho*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
%RMSprop/w2v_embedding_cbow/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*6
shared_name'%RMSprop/w2v_embedding_cbow/kernel/rms
?
9RMSprop/w2v_embedding_cbow/kernel/rms/Read/ReadVariableOpReadVariableOp%RMSprop/w2v_embedding_cbow/kernel/rms* 
_output_shapes
:
??*
dtype0
?
#RMSprop/w2v_embedding_cbow/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#RMSprop/w2v_embedding_cbow/bias/rms
?
7RMSprop/w2v_embedding_cbow/bias/rms/Read/ReadVariableOpReadVariableOp#RMSprop/w2v_embedding_cbow/bias/rms*
_output_shapes	
:?*
dtype0
?
RMSprop/dense/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*)
shared_nameRMSprop/dense/kernel/rms
?
,RMSprop/dense/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense/kernel/rms* 
_output_shapes
:
??*
dtype0
?
RMSprop/dense/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*'
shared_nameRMSprop/dense/bias/rms
~
*RMSprop/dense/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense/bias/rms*
_output_shapes	
:?*
dtype0

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
	optimizer
trainable_variables
	variables
regularization_losses
		keras_api


signatures
 
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
R
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
k
iter
	decay
learning_rate
momentum
rho	rms?	rms@	rmsA	rmsB

0
1
2
3

0
1
2
3
 
?

 layers
!metrics
"layer_metrics
#non_trainable_variables
$layer_regularization_losses
trainable_variables
	variables
regularization_losses
 
ec
VARIABLE_VALUEw2v_embedding_cbow/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEw2v_embedding_cbow/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?

%layers
&metrics
'layer_metrics
(non_trainable_variables
)layer_regularization_losses
trainable_variables
	variables
regularization_losses
 
 
 
?

*layers
+metrics
,layer_metrics
-non_trainable_variables
.layer_regularization_losses
trainable_variables
	variables
regularization_losses
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?

/layers
0metrics
1layer_metrics
2non_trainable_variables
3layer_regularization_losses
trainable_variables
	variables
regularization_losses
KI
VARIABLE_VALUERMSprop/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUERMSprop/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUERMSprop/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUERMSprop/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUERMSprop/rho(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
3

40
51
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	6total
	7count
8	variables
9	keras_api
D
	:total
	;count
<
_fn_kwargs
=	variables
>	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

60
71

8	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

:0
;1

=	variables
??
VARIABLE_VALUE%RMSprop/w2v_embedding_cbow/kernel/rmsTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#RMSprop/w2v_embedding_cbow/bias/rmsRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/dense/kernel/rmsTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUERMSprop/dense/bias/rmsRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
|
serving_default_input_1Placeholder*(
_output_shapes
:??????????*
dtype0*
shape:??????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1w2v_embedding_cbow/kernelw2v_embedding_cbow/biasdense/kernel
dense/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *.
f)R'
%__inference_signature_wrapper_4492709
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename-w2v_embedding_cbow/kernel/Read/ReadVariableOp+w2v_embedding_cbow/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp RMSprop/iter/Read/ReadVariableOp!RMSprop/decay/Read/ReadVariableOp)RMSprop/learning_rate/Read/ReadVariableOp$RMSprop/momentum/Read/ReadVariableOpRMSprop/rho/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp9RMSprop/w2v_embedding_cbow/kernel/rms/Read/ReadVariableOp7RMSprop/w2v_embedding_cbow/bias/rms/Read/ReadVariableOp,RMSprop/dense/kernel/rms/Read/ReadVariableOp*RMSprop/dense/bias/rms/Read/ReadVariableOpConst*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *)
f$R"
 __inference__traced_save_4492897
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamew2v_embedding_cbow/kernelw2v_embedding_cbow/biasdense/kernel
dense/biasRMSprop/iterRMSprop/decayRMSprop/learning_rateRMSprop/momentumRMSprop/rhototalcounttotal_1count_1%RMSprop/w2v_embedding_cbow/kernel/rms#RMSprop/w2v_embedding_cbow/bias/rmsRMSprop/dense/kernel/rmsRMSprop/dense/bias/rms*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *,
f'R%
#__inference__traced_restore_4492958??
?
`
D__inference_average_layer_call_and_return_conditional_losses_4492803

inputs
identity[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
	truediv/yl
truedivRealDivinputstruediv/y:output:0*
T0*(
_output_shapes
:??????????2	
truediv`
IdentityIdentitytruediv:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
B__inference_dense_layer_call_and_return_conditional_losses_4492823

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddb
SoftmaxSoftmaxBiasAdd:output:0*
T0*(
_output_shapes
:??????????2	
Softmaxm
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
B__inference_model_layer_call_and_return_conditional_losses_4492754

inputsE
1w2v_embedding_cbow_matmul_readvariableop_resource:
??A
2w2v_embedding_cbow_biasadd_readvariableop_resource:	?8
$dense_matmul_readvariableop_resource:
??4
%dense_biasadd_readvariableop_resource:	?
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?)w2v_embedding_cbow/BiasAdd/ReadVariableOp?(w2v_embedding_cbow/MatMul/ReadVariableOp?
(w2v_embedding_cbow/MatMul/ReadVariableOpReadVariableOp1w2v_embedding_cbow_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02*
(w2v_embedding_cbow/MatMul/ReadVariableOp?
w2v_embedding_cbow/MatMulMatMulinputs0w2v_embedding_cbow/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
w2v_embedding_cbow/MatMul?
)w2v_embedding_cbow/BiasAdd/ReadVariableOpReadVariableOp2w2v_embedding_cbow_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)w2v_embedding_cbow/BiasAdd/ReadVariableOp?
w2v_embedding_cbow/BiasAddBiasAdd#w2v_embedding_cbow/MatMul:product:01w2v_embedding_cbow/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
w2v_embedding_cbow/BiasAddk
average/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
average/truediv/y?
average/truedivRealDiv#w2v_embedding_cbow/BiasAdd:output:0average/truediv/y:output:0*
T0*(
_output_shapes
:??????????2
average/truediv?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulaverage/truediv:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/BiasAddt
dense/SoftmaxSoftmaxdense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense/Softmaxs
IdentityIdentitydense/Softmax:softmax:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*^w2v_embedding_cbow/BiasAdd/ReadVariableOp)^w2v_embedding_cbow/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2V
)w2v_embedding_cbow/BiasAdd/ReadVariableOp)w2v_embedding_cbow/BiasAdd/ReadVariableOp2T
(w2v_embedding_cbow/MatMul/ReadVariableOp(w2v_embedding_cbow/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?-
?
 __inference__traced_save_4492897
file_prefix8
4savev2_w2v_embedding_cbow_kernel_read_readvariableop6
2savev2_w2v_embedding_cbow_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop+
'savev2_rmsprop_iter_read_readvariableop	,
(savev2_rmsprop_decay_read_readvariableop4
0savev2_rmsprop_learning_rate_read_readvariableop/
+savev2_rmsprop_momentum_read_readvariableop*
&savev2_rmsprop_rho_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableopD
@savev2_rmsprop_w2v_embedding_cbow_kernel_rms_read_readvariableopB
>savev2_rmsprop_w2v_embedding_cbow_bias_rms_read_readvariableop7
3savev2_rmsprop_dense_kernel_rms_read_readvariableop5
1savev2_rmsprop_dense_bias_rms_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?	
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*7
value.B,B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:04savev2_w2v_embedding_cbow_kernel_read_readvariableop2savev2_w2v_embedding_cbow_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop'savev2_rmsprop_iter_read_readvariableop(savev2_rmsprop_decay_read_readvariableop0savev2_rmsprop_learning_rate_read_readvariableop+savev2_rmsprop_momentum_read_readvariableop&savev2_rmsprop_rho_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop@savev2_rmsprop_w2v_embedding_cbow_kernel_rms_read_readvariableop>savev2_rmsprop_w2v_embedding_cbow_bias_rms_read_readvariableop3savev2_rmsprop_dense_kernel_rms_read_readvariableop1savev2_rmsprop_dense_bias_rms_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 * 
dtypes
2	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*u
_input_shapesd
b: :
??:?:
??:?: : : : : : : : : :
??:?:
??:?: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:

_output_shapes
: 
?K
?

#__inference__traced_restore_4492958
file_prefix>
*assignvariableop_w2v_embedding_cbow_kernel:
??9
*assignvariableop_1_w2v_embedding_cbow_bias:	?3
assignvariableop_2_dense_kernel:
??,
assignvariableop_3_dense_bias:	?)
assignvariableop_4_rmsprop_iter:	 *
 assignvariableop_5_rmsprop_decay: 2
(assignvariableop_6_rmsprop_learning_rate: -
#assignvariableop_7_rmsprop_momentum: (
assignvariableop_8_rmsprop_rho: "
assignvariableop_9_total: #
assignvariableop_10_count: %
assignvariableop_11_total_1: %
assignvariableop_12_count_1: M
9assignvariableop_13_rmsprop_w2v_embedding_cbow_kernel_rms:
??F
7assignvariableop_14_rmsprop_w2v_embedding_cbow_bias_rms:	?@
,assignvariableop_15_rmsprop_dense_kernel_rms:
??9
*assignvariableop_16_rmsprop_dense_bias_rms:	?
identity_18??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?	
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*7
value.B,B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*\
_output_shapesJ
H::::::::::::::::::* 
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp*assignvariableop_w2v_embedding_cbow_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp*assignvariableop_1_w2v_embedding_cbow_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_dense_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_rmsprop_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp assignvariableop_5_rmsprop_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp(assignvariableop_6_rmsprop_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp#assignvariableop_7_rmsprop_momentumIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_rmsprop_rhoIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_totalIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpassignvariableop_10_countIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_total_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_count_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp9assignvariableop_13_rmsprop_w2v_embedding_cbow_kernel_rmsIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp7assignvariableop_14_rmsprop_w2v_embedding_cbow_bias_rmsIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp,assignvariableop_15_rmsprop_dense_kernel_rmsIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp*assignvariableop_16_rmsprop_dense_bias_rmsIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_169
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_17Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_17f
Identity_18IdentityIdentity_17:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_18?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_18Identity_18:output:0*7
_input_shapes&
$: : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
"__inference__wrapped_model_4492518
input_1K
7model_w2v_embedding_cbow_matmul_readvariableop_resource:
??G
8model_w2v_embedding_cbow_biasadd_readvariableop_resource:	?>
*model_dense_matmul_readvariableop_resource:
??:
+model_dense_biasadd_readvariableop_resource:	?
identity??"model/dense/BiasAdd/ReadVariableOp?!model/dense/MatMul/ReadVariableOp?/model/w2v_embedding_cbow/BiasAdd/ReadVariableOp?.model/w2v_embedding_cbow/MatMul/ReadVariableOp?
.model/w2v_embedding_cbow/MatMul/ReadVariableOpReadVariableOp7model_w2v_embedding_cbow_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype020
.model/w2v_embedding_cbow/MatMul/ReadVariableOp?
model/w2v_embedding_cbow/MatMulMatMulinput_16model/w2v_embedding_cbow/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
model/w2v_embedding_cbow/MatMul?
/model/w2v_embedding_cbow/BiasAdd/ReadVariableOpReadVariableOp8model_w2v_embedding_cbow_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype021
/model/w2v_embedding_cbow/BiasAdd/ReadVariableOp?
 model/w2v_embedding_cbow/BiasAddBiasAdd)model/w2v_embedding_cbow/MatMul:product:07model/w2v_embedding_cbow/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 model/w2v_embedding_cbow/BiasAddw
model/average/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
model/average/truediv/y?
model/average/truedivRealDiv)model/w2v_embedding_cbow/BiasAdd:output:0 model/average/truediv/y:output:0*
T0*(
_output_shapes
:??????????2
model/average/truediv?
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02#
!model/dense/MatMul/ReadVariableOp?
model/dense/MatMulMatMulmodel/average/truediv:z:0)model/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/dense/MatMul?
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"model/dense/BiasAdd/ReadVariableOp?
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/dense/BiasAdd?
model/dense/SoftmaxSoftmaxmodel/dense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model/dense/Softmaxy
IdentityIdentitymodel/dense/Softmax:softmax:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp0^model/w2v_embedding_cbow/BiasAdd/ReadVariableOp/^model/w2v_embedding_cbow/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2b
/model/w2v_embedding_cbow/BiasAdd/ReadVariableOp/model/w2v_embedding_cbow/BiasAdd/ReadVariableOp2`
.model/w2v_embedding_cbow/MatMul/ReadVariableOp.model/w2v_embedding_cbow/MatMul/ReadVariableOp:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?
E
)__inference_average_layer_call_fn_4492797

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_average_layer_call_and_return_conditional_losses_44925472
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
'__inference_dense_layer_call_fn_4492812

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_44925602
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
B__inference_dense_layer_call_and_return_conditional_losses_4492560

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddb
SoftmaxSoftmaxBiasAdd:output:0*
T0*(
_output_shapes
:??????????2	
Softmaxm
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
'__inference_model_layer_call_fn_4492658
input_1
unknown:
??
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_44926342
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?
?
4__inference_w2v_embedding_cbow_layer_call_fn_4492782

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *X
fSRQ
O__inference_w2v_embedding_cbow_layer_call_and_return_conditional_losses_44925352
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
B__inference_model_layer_call_and_return_conditional_losses_4492688
input_1.
w2v_embedding_cbow_4492676:
??)
w2v_embedding_cbow_4492678:	?!
dense_4492682:
??
dense_4492684:	?
identity??dense/StatefulPartitionedCall?*w2v_embedding_cbow/StatefulPartitionedCall?
*w2v_embedding_cbow/StatefulPartitionedCallStatefulPartitionedCallinput_1w2v_embedding_cbow_4492676w2v_embedding_cbow_4492678*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *X
fSRQ
O__inference_w2v_embedding_cbow_layer_call_and_return_conditional_losses_44925352,
*w2v_embedding_cbow/StatefulPartitionedCall?
average/PartitionedCallPartitionedCall3w2v_embedding_cbow/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_average_layer_call_and_return_conditional_losses_44925472
average/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall average/PartitionedCall:output:0dense_4492682dense_4492684*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_44925602
dense/StatefulPartitionedCall?
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp^dense/StatefulPartitionedCall+^w2v_embedding_cbow/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2X
*w2v_embedding_cbow/StatefulPartitionedCall*w2v_embedding_cbow/StatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?
?
B__inference_model_layer_call_and_return_conditional_losses_4492673
input_1.
w2v_embedding_cbow_4492661:
??)
w2v_embedding_cbow_4492663:	?!
dense_4492667:
??
dense_4492669:	?
identity??dense/StatefulPartitionedCall?*w2v_embedding_cbow/StatefulPartitionedCall?
*w2v_embedding_cbow/StatefulPartitionedCallStatefulPartitionedCallinput_1w2v_embedding_cbow_4492661w2v_embedding_cbow_4492663*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *X
fSRQ
O__inference_w2v_embedding_cbow_layer_call_and_return_conditional_losses_44925352,
*w2v_embedding_cbow/StatefulPartitionedCall?
average/PartitionedCallPartitionedCall3w2v_embedding_cbow/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_average_layer_call_and_return_conditional_losses_44925472
average/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall average/PartitionedCall:output:0dense_4492667dense_4492669*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_44925602
dense/StatefulPartitionedCall?
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp^dense/StatefulPartitionedCall+^w2v_embedding_cbow/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2X
*w2v_embedding_cbow/StatefulPartitionedCall*w2v_embedding_cbow/StatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?

?
O__inference_w2v_embedding_cbow_layer_call_and_return_conditional_losses_4492792

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddl
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
B__inference_model_layer_call_and_return_conditional_losses_4492773

inputsE
1w2v_embedding_cbow_matmul_readvariableop_resource:
??A
2w2v_embedding_cbow_biasadd_readvariableop_resource:	?8
$dense_matmul_readvariableop_resource:
??4
%dense_biasadd_readvariableop_resource:	?
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?)w2v_embedding_cbow/BiasAdd/ReadVariableOp?(w2v_embedding_cbow/MatMul/ReadVariableOp?
(w2v_embedding_cbow/MatMul/ReadVariableOpReadVariableOp1w2v_embedding_cbow_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02*
(w2v_embedding_cbow/MatMul/ReadVariableOp?
w2v_embedding_cbow/MatMulMatMulinputs0w2v_embedding_cbow/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
w2v_embedding_cbow/MatMul?
)w2v_embedding_cbow/BiasAdd/ReadVariableOpReadVariableOp2w2v_embedding_cbow_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)w2v_embedding_cbow/BiasAdd/ReadVariableOp?
w2v_embedding_cbow/BiasAddBiasAdd#w2v_embedding_cbow/MatMul:product:01w2v_embedding_cbow/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
w2v_embedding_cbow/BiasAddk
average/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
average/truediv/y?
average/truedivRealDiv#w2v_embedding_cbow/BiasAdd:output:0average/truediv/y:output:0*
T0*(
_output_shapes
:??????????2
average/truediv?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulaverage/truediv:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/BiasAddt
dense/SoftmaxSoftmaxdense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense/Softmaxs
IdentityIdentitydense/Softmax:softmax:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*^w2v_embedding_cbow/BiasAdd/ReadVariableOp)^w2v_embedding_cbow/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2V
)w2v_embedding_cbow/BiasAdd/ReadVariableOp)w2v_embedding_cbow/BiasAdd/ReadVariableOp2T
(w2v_embedding_cbow/MatMul/ReadVariableOp(w2v_embedding_cbow/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
'__inference_model_layer_call_fn_4492735

inputs
unknown:
??
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_44926342
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
`
D__inference_average_layer_call_and_return_conditional_losses_4492547

inputs
identity[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
	truediv/yl
truedivRealDivinputstruediv/y:output:0*
T0*(
_output_shapes
:??????????2	
truediv`
IdentityIdentitytruediv:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
O__inference_w2v_embedding_cbow_layer_call_and_return_conditional_losses_4492535

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddl
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
B__inference_model_layer_call_and_return_conditional_losses_4492567

inputs.
w2v_embedding_cbow_4492536:
??)
w2v_embedding_cbow_4492538:	?!
dense_4492561:
??
dense_4492563:	?
identity??dense/StatefulPartitionedCall?*w2v_embedding_cbow/StatefulPartitionedCall?
*w2v_embedding_cbow/StatefulPartitionedCallStatefulPartitionedCallinputsw2v_embedding_cbow_4492536w2v_embedding_cbow_4492538*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *X
fSRQ
O__inference_w2v_embedding_cbow_layer_call_and_return_conditional_losses_44925352,
*w2v_embedding_cbow/StatefulPartitionedCall?
average/PartitionedCallPartitionedCall3w2v_embedding_cbow/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_average_layer_call_and_return_conditional_losses_44925472
average/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall average/PartitionedCall:output:0dense_4492561dense_4492563*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_44925602
dense/StatefulPartitionedCall?
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp^dense/StatefulPartitionedCall+^w2v_embedding_cbow/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2X
*w2v_embedding_cbow/StatefulPartitionedCall*w2v_embedding_cbow/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
B__inference_model_layer_call_and_return_conditional_losses_4492634

inputs.
w2v_embedding_cbow_4492622:
??)
w2v_embedding_cbow_4492624:	?!
dense_4492628:
??
dense_4492630:	?
identity??dense/StatefulPartitionedCall?*w2v_embedding_cbow/StatefulPartitionedCall?
*w2v_embedding_cbow/StatefulPartitionedCallStatefulPartitionedCallinputsw2v_embedding_cbow_4492622w2v_embedding_cbow_4492624*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *X
fSRQ
O__inference_w2v_embedding_cbow_layer_call_and_return_conditional_losses_44925352,
*w2v_embedding_cbow/StatefulPartitionedCall?
average/PartitionedCallPartitionedCall3w2v_embedding_cbow/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_average_layer_call_and_return_conditional_losses_44925472
average/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall average/PartitionedCall:output:0dense_4492628dense_4492630*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_44925602
dense/StatefulPartitionedCall?
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp^dense/StatefulPartitionedCall+^w2v_embedding_cbow/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2X
*w2v_embedding_cbow/StatefulPartitionedCall*w2v_embedding_cbow/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
'__inference_model_layer_call_fn_4492722

inputs
unknown:
??
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_44925672
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
'__inference_model_layer_call_fn_4492578
input_1
unknown:
??
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_44925672
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?
?
%__inference_signature_wrapper_4492709
input_1
unknown:
??
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *+
f&R$
"__inference__wrapped_model_44925182
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
<
input_11
serving_default_input_1:0??????????:
dense1
StatefulPartitionedCall:0??????????tensorflow/serving/predict:?K
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
	optimizer
trainable_variables
	variables
regularization_losses
		keras_api


signatures
C_default_save_signature
D__call__
*E&call_and_return_all_conditional_losses"
_tf_keras_network
"
_tf_keras_input_layer
?

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
F__call__
*G&call_and_return_all_conditional_losses"
_tf_keras_layer
?
trainable_variables
	variables
regularization_losses
	keras_api
H__call__
*I&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
J__call__
*K&call_and_return_all_conditional_losses"
_tf_keras_layer
~
iter
	decay
learning_rate
momentum
rho	rms?	rms@	rmsA	rmsB"
	optimizer
<
0
1
2
3"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
?

 layers
!metrics
"layer_metrics
#non_trainable_variables
$layer_regularization_losses
trainable_variables
	variables
regularization_losses
D__call__
C_default_save_signature
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
,
Lserving_default"
signature_map
-:+
??2w2v_embedding_cbow/kernel
&:$?2w2v_embedding_cbow/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?

%layers
&metrics
'layer_metrics
(non_trainable_variables
)layer_regularization_losses
trainable_variables
	variables
regularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

*layers
+metrics
,layer_metrics
-non_trainable_variables
.layer_regularization_losses
trainable_variables
	variables
regularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses"
_generic_user_object
 :
??2dense/kernel
:?2
dense/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?

/layers
0metrics
1layer_metrics
2non_trainable_variables
3layer_regularization_losses
trainable_variables
	variables
regularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses"
_generic_user_object
:	 (2RMSprop/iter
: (2RMSprop/decay
: (2RMSprop/learning_rate
: (2RMSprop/momentum
: (2RMSprop/rho
<
0
1
2
3"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
N
	6total
	7count
8	variables
9	keras_api"
_tf_keras_metric
^
	:total
	;count
<
_fn_kwargs
=	variables
>	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
60
71"
trackable_list_wrapper
-
8	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
:0
;1"
trackable_list_wrapper
-
=	variables"
_generic_user_object
7:5
??2%RMSprop/w2v_embedding_cbow/kernel/rms
0:.?2#RMSprop/w2v_embedding_cbow/bias/rms
*:(
??2RMSprop/dense/kernel/rms
#:!?2RMSprop/dense/bias/rms
?B?
"__inference__wrapped_model_4492518input_1"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_model_layer_call_fn_4492578
'__inference_model_layer_call_fn_4492722
'__inference_model_layer_call_fn_4492735
'__inference_model_layer_call_fn_4492658?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_model_layer_call_and_return_conditional_losses_4492754
B__inference_model_layer_call_and_return_conditional_losses_4492773
B__inference_model_layer_call_and_return_conditional_losses_4492673
B__inference_model_layer_call_and_return_conditional_losses_4492688?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
4__inference_w2v_embedding_cbow_layer_call_fn_4492782?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
O__inference_w2v_embedding_cbow_layer_call_and_return_conditional_losses_4492792?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_average_layer_call_fn_4492797?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_average_layer_call_and_return_conditional_losses_4492803?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense_layer_call_fn_4492812?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_layer_call_and_return_conditional_losses_4492823?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
%__inference_signature_wrapper_4492709input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
"__inference__wrapped_model_4492518i1?.
'?$
"?
input_1??????????
? ".?+
)
dense ?
dense???????????
D__inference_average_layer_call_and_return_conditional_losses_4492803Z0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? z
)__inference_average_layer_call_fn_4492797M0?-
&?#
!?
inputs??????????
? "????????????
B__inference_dense_layer_call_and_return_conditional_losses_4492823^0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? |
'__inference_dense_layer_call_fn_4492812Q0?-
&?#
!?
inputs??????????
? "????????????
B__inference_model_layer_call_and_return_conditional_losses_4492673i9?6
/?,
"?
input_1??????????
p 

 
? "&?#
?
0??????????
? ?
B__inference_model_layer_call_and_return_conditional_losses_4492688i9?6
/?,
"?
input_1??????????
p

 
? "&?#
?
0??????????
? ?
B__inference_model_layer_call_and_return_conditional_losses_4492754h8?5
.?+
!?
inputs??????????
p 

 
? "&?#
?
0??????????
? ?
B__inference_model_layer_call_and_return_conditional_losses_4492773h8?5
.?+
!?
inputs??????????
p

 
? "&?#
?
0??????????
? ?
'__inference_model_layer_call_fn_4492578\9?6
/?,
"?
input_1??????????
p 

 
? "????????????
'__inference_model_layer_call_fn_4492658\9?6
/?,
"?
input_1??????????
p

 
? "????????????
'__inference_model_layer_call_fn_4492722[8?5
.?+
!?
inputs??????????
p 

 
? "????????????
'__inference_model_layer_call_fn_4492735[8?5
.?+
!?
inputs??????????
p

 
? "????????????
%__inference_signature_wrapper_4492709t<?9
? 
2?/
-
input_1"?
input_1??????????".?+
)
dense ?
dense???????????
O__inference_w2v_embedding_cbow_layer_call_and_return_conditional_losses_4492792^0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
4__inference_w2v_embedding_cbow_layer_call_fn_4492782Q0?-
&?#
!?
inputs??????????
? "???????????