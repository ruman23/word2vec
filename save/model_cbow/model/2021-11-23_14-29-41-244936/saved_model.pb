??
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
??**
shared_namew2v_embedding_cbow/kernel
?
-w2v_embedding_cbow/kernel/Read/ReadVariableOpReadVariableOpw2v_embedding_cbow/kernel* 
_output_shapes
:
??*
dtype0
?
w2v_embedding_cbow/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*(
shared_namew2v_embedding_cbow/bias
?
+w2v_embedding_cbow/bias/Read/ReadVariableOpReadVariableOpw2v_embedding_cbow/bias*
_output_shapes	
:?*
dtype0
|
dense_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_12/kernel
u
#dense_12/kernel/Read/ReadVariableOpReadVariableOpdense_12/kernel* 
_output_shapes
:
??*
dtype0
s
dense_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_12/bias
l
!dense_12/bias/Read/ReadVariableOpReadVariableOpdense_12/bias*
_output_shapes	
:?*
dtype0
d
SGD/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
SGD/iter
]
SGD/iter/Read/ReadVariableOpReadVariableOpSGD/iter*
_output_shapes
: *
dtype0	
f
	SGD/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	SGD/decay
_
SGD/decay/Read/ReadVariableOpReadVariableOp	SGD/decay*
_output_shapes
: *
dtype0
v
SGD/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameSGD/learning_rate
o
%SGD/learning_rate/Read/ReadVariableOpReadVariableOpSGD/learning_rate*
_output_shapes
: *
dtype0
l
SGD/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameSGD/momentum
e
 SGD/momentum/Read/ReadVariableOpReadVariableOpSGD/momentum*
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

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
	optimizer
	variables
trainable_variables
regularization_losses
		keras_api


signatures
 
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
6
iter
	decay
learning_rate
momentum

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

layers
 layer_metrics
	variables
trainable_variables
regularization_losses
!non_trainable_variables
"metrics
#layer_regularization_losses
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

$layers
%layer_metrics
	variables
trainable_variables
regularization_losses
&non_trainable_variables
'metrics
(layer_regularization_losses
 
 
 
?

)layers
*layer_metrics
	variables
trainable_variables
regularization_losses
+non_trainable_variables
,metrics
-layer_regularization_losses
[Y
VARIABLE_VALUEdense_12/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_12/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?

.layers
/layer_metrics
	variables
trainable_variables
regularization_losses
0non_trainable_variables
1metrics
2layer_regularization_losses
GE
VARIABLE_VALUESGD/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUE	SGD/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUESGD/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUESGD/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
3
 
 

30
41
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
	5total
	6count
7	variables
8	keras_api
D
	9total
	:count
;
_fn_kwargs
<	variables
=	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

50
61

7	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

90
:1

<	variables
}
serving_default_input_13Placeholder*(
_output_shapes
:??????????*
dtype0*
shape:??????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_13w2v_embedding_cbow/kernelw2v_embedding_cbow/biasdense_12/kerneldense_12/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? */
f*R(
&__inference_signature_wrapper_11958735
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename-w2v_embedding_cbow/kernel/Read/ReadVariableOp+w2v_embedding_cbow/bias/Read/ReadVariableOp#dense_12/kernel/Read/ReadVariableOp!dense_12/bias/Read/ReadVariableOpSGD/iter/Read/ReadVariableOpSGD/decay/Read/ReadVariableOp%SGD/learning_rate/Read/ReadVariableOp SGD/momentum/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOpConst*
Tin
2	*
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
GPU2 *0J 8? **
f%R#
!__inference__traced_save_11958908
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamew2v_embedding_cbow/kernelw2v_embedding_cbow/biasdense_12/kerneldense_12/biasSGD/iter	SGD/decaySGD/learning_rateSGD/momentumtotalcounttotal_1count_1*
Tin
2*
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
GPU2 *0J 8? *-
f(R&
$__inference__traced_restore_11958954??
?
?
F__inference_model_12_layer_call_and_return_conditional_losses_11958662

inputs/
w2v_embedding_cbow_11958650:
??*
w2v_embedding_cbow_11958652:	?%
dense_12_11958656:
?? 
dense_12_11958658:	?
identity?? dense_12/StatefulPartitionedCall?*w2v_embedding_cbow/StatefulPartitionedCall?
*w2v_embedding_cbow/StatefulPartitionedCallStatefulPartitionedCallinputsw2v_embedding_cbow_11958650w2v_embedding_cbow_11958652*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_w2v_embedding_cbow_layer_call_and_return_conditional_losses_119585632,
*w2v_embedding_cbow/StatefulPartitionedCall?
average_6/PartitionedCallPartitionedCall3w2v_embedding_cbow/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_average_6_layer_call_and_return_conditional_losses_119585752
average_6/PartitionedCall?
 dense_12/StatefulPartitionedCallStatefulPartitionedCall"average_6/PartitionedCall:output:0dense_12_11958656dense_12_11958658*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_12_layer_call_and_return_conditional_losses_119585882"
 dense_12/StatefulPartitionedCall?
IdentityIdentity)dense_12/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp!^dense_12/StatefulPartitionedCall+^w2v_embedding_cbow/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2X
*w2v_embedding_cbow/StatefulPartitionedCall*w2v_embedding_cbow/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?6
?
$__inference__traced_restore_11958954
file_prefix>
*assignvariableop_w2v_embedding_cbow_kernel:
??9
*assignvariableop_1_w2v_embedding_cbow_bias:	?6
"assignvariableop_2_dense_12_kernel:
??/
 assignvariableop_3_dense_12_bias:	?%
assignvariableop_4_sgd_iter:	 &
assignvariableop_5_sgd_decay: .
$assignvariableop_6_sgd_learning_rate: )
assignvariableop_7_sgd_momentum: "
assignvariableop_8_total: "
assignvariableop_9_count: %
assignvariableop_10_total_1: %
assignvariableop_11_count_1: 
identity_13??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*H
_output_shapes6
4:::::::::::::*
dtypes
2	2
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
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_12_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_12_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_sgd_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_sgd_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp$assignvariableop_6_sgd_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_sgd_momentumIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_totalIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_countIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpassignvariableop_10_total_1Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_count_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_119
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_12Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_12f
Identity_13IdentityIdentity_12:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_13?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_13Identity_13:output:0*-
_input_shapes
: : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112(
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
?
?
F__inference_model_12_layer_call_and_return_conditional_losses_11958716
input_13/
w2v_embedding_cbow_11958704:
??*
w2v_embedding_cbow_11958706:	?%
dense_12_11958710:
?? 
dense_12_11958712:	?
identity?? dense_12/StatefulPartitionedCall?*w2v_embedding_cbow/StatefulPartitionedCall?
*w2v_embedding_cbow/StatefulPartitionedCallStatefulPartitionedCallinput_13w2v_embedding_cbow_11958704w2v_embedding_cbow_11958706*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_w2v_embedding_cbow_layer_call_and_return_conditional_losses_119585632,
*w2v_embedding_cbow/StatefulPartitionedCall?
average_6/PartitionedCallPartitionedCall3w2v_embedding_cbow/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_average_6_layer_call_and_return_conditional_losses_119585752
average_6/PartitionedCall?
 dense_12/StatefulPartitionedCallStatefulPartitionedCall"average_6/PartitionedCall:output:0dense_12_11958710dense_12_11958712*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_12_layer_call_and_return_conditional_losses_119585882"
 dense_12/StatefulPartitionedCall?
IdentityIdentity)dense_12/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp!^dense_12/StatefulPartitionedCall+^w2v_embedding_cbow/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2X
*w2v_embedding_cbow/StatefulPartitionedCall*w2v_embedding_cbow/StatefulPartitionedCall:R N
(
_output_shapes
:??????????
"
_user_specified_name
input_13
?$
?
!__inference__traced_save_11958908
file_prefix8
4savev2_w2v_embedding_cbow_kernel_read_readvariableop6
2savev2_w2v_embedding_cbow_bias_read_readvariableop.
*savev2_dense_12_kernel_read_readvariableop,
(savev2_dense_12_bias_read_readvariableop'
#savev2_sgd_iter_read_readvariableop	(
$savev2_sgd_decay_read_readvariableop0
,savev2_sgd_learning_rate_read_readvariableop+
'savev2_sgd_momentum_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop
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
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:04savev2_w2v_embedding_cbow_kernel_read_readvariableop2savev2_w2v_embedding_cbow_bias_read_readvariableop*savev2_dense_12_kernel_read_readvariableop(savev2_dense_12_bias_read_readvariableop#savev2_sgd_iter_read_readvariableop$savev2_sgd_decay_read_readvariableop,savev2_sgd_learning_rate_read_readvariableop'savev2_sgd_momentum_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	2
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

identity_1Identity_1:output:0*M
_input_shapes<
:: :
??:?:
??:?: : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:
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
: 
?
c
G__inference_average_6_layer_call_and_return_conditional_losses_11958575

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
:??????????2	
truediv`
IdentityIdentitytruediv:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
F__inference_dense_12_layer_call_and_return_conditional_losses_11958840

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddb
SoftmaxSoftmaxBiasAdd:output:0*
T0*(
_output_shapes
:??????????2	
Softmaxm
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
#__inference__wrapped_model_11958546
input_13N
:model_12_w2v_embedding_cbow_matmul_readvariableop_resource:
??J
;model_12_w2v_embedding_cbow_biasadd_readvariableop_resource:	?D
0model_12_dense_12_matmul_readvariableop_resource:
??@
1model_12_dense_12_biasadd_readvariableop_resource:	?
identity??(model_12/dense_12/BiasAdd/ReadVariableOp?'model_12/dense_12/MatMul/ReadVariableOp?2model_12/w2v_embedding_cbow/BiasAdd/ReadVariableOp?1model_12/w2v_embedding_cbow/MatMul/ReadVariableOp?
1model_12/w2v_embedding_cbow/MatMul/ReadVariableOpReadVariableOp:model_12_w2v_embedding_cbow_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype023
1model_12/w2v_embedding_cbow/MatMul/ReadVariableOp?
"model_12/w2v_embedding_cbow/MatMulMatMulinput_139model_12/w2v_embedding_cbow/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2$
"model_12/w2v_embedding_cbow/MatMul?
2model_12/w2v_embedding_cbow/BiasAdd/ReadVariableOpReadVariableOp;model_12_w2v_embedding_cbow_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype024
2model_12/w2v_embedding_cbow/BiasAdd/ReadVariableOp?
#model_12/w2v_embedding_cbow/BiasAddBiasAdd,model_12/w2v_embedding_cbow/MatMul:product:0:model_12/w2v_embedding_cbow/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2%
#model_12/w2v_embedding_cbow/BiasAdd?
model_12/average_6/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
model_12/average_6/truediv/y?
model_12/average_6/truedivRealDiv,model_12/w2v_embedding_cbow/BiasAdd:output:0%model_12/average_6/truediv/y:output:0*
T0*(
_output_shapes
:??????????2
model_12/average_6/truediv?
'model_12/dense_12/MatMul/ReadVariableOpReadVariableOp0model_12_dense_12_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02)
'model_12/dense_12/MatMul/ReadVariableOp?
model_12/dense_12/MatMulMatMulmodel_12/average_6/truediv:z:0/model_12/dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_12/dense_12/MatMul?
(model_12/dense_12/BiasAdd/ReadVariableOpReadVariableOp1model_12_dense_12_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(model_12/dense_12/BiasAdd/ReadVariableOp?
model_12/dense_12/BiasAddBiasAdd"model_12/dense_12/MatMul:product:00model_12/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_12/dense_12/BiasAdd?
model_12/dense_12/SoftmaxSoftmax"model_12/dense_12/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model_12/dense_12/Softmax
IdentityIdentity#model_12/dense_12/Softmax:softmax:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp)^model_12/dense_12/BiasAdd/ReadVariableOp(^model_12/dense_12/MatMul/ReadVariableOp3^model_12/w2v_embedding_cbow/BiasAdd/ReadVariableOp2^model_12/w2v_embedding_cbow/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2T
(model_12/dense_12/BiasAdd/ReadVariableOp(model_12/dense_12/BiasAdd/ReadVariableOp2R
'model_12/dense_12/MatMul/ReadVariableOp'model_12/dense_12/MatMul/ReadVariableOp2h
2model_12/w2v_embedding_cbow/BiasAdd/ReadVariableOp2model_12/w2v_embedding_cbow/BiasAdd/ReadVariableOp2f
1model_12/w2v_embedding_cbow/MatMul/ReadVariableOp1model_12/w2v_embedding_cbow/MatMul/ReadVariableOp:R N
(
_output_shapes
:??????????
"
_user_specified_name
input_13
?
?
F__inference_model_12_layer_call_and_return_conditional_losses_11958773

inputsE
1w2v_embedding_cbow_matmul_readvariableop_resource:
??A
2w2v_embedding_cbow_biasadd_readvariableop_resource:	?;
'dense_12_matmul_readvariableop_resource:
??7
(dense_12_biasadd_readvariableop_resource:	?
identity??dense_12/BiasAdd/ReadVariableOp?dense_12/MatMul/ReadVariableOp?)w2v_embedding_cbow/BiasAdd/ReadVariableOp?(w2v_embedding_cbow/MatMul/ReadVariableOp?
(w2v_embedding_cbow/MatMul/ReadVariableOpReadVariableOp1w2v_embedding_cbow_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02*
(w2v_embedding_cbow/MatMul/ReadVariableOp?
w2v_embedding_cbow/MatMulMatMulinputs0w2v_embedding_cbow/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
w2v_embedding_cbow/MatMul?
)w2v_embedding_cbow/BiasAdd/ReadVariableOpReadVariableOp2w2v_embedding_cbow_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)w2v_embedding_cbow/BiasAdd/ReadVariableOp?
w2v_embedding_cbow/BiasAddBiasAdd#w2v_embedding_cbow/MatMul:product:01w2v_embedding_cbow/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
w2v_embedding_cbow/BiasAddo
average_6/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
average_6/truediv/y?
average_6/truedivRealDiv#w2v_embedding_cbow/BiasAdd:output:0average_6/truediv/y:output:0*
T0*(
_output_shapes
:??????????2
average_6/truediv?
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_12/MatMul/ReadVariableOp?
dense_12/MatMulMatMulaverage_6/truediv:z:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_12/MatMul?
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_12/BiasAdd/ReadVariableOp?
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_12/BiasAdd}
dense_12/SoftmaxSoftmaxdense_12/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_12/Softmaxv
IdentityIdentitydense_12/Softmax:softmax:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp*^w2v_embedding_cbow/BiasAdd/ReadVariableOp)^w2v_embedding_cbow/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2V
)w2v_embedding_cbow/BiasAdd/ReadVariableOp)w2v_embedding_cbow/BiasAdd/ReadVariableOp2T
(w2v_embedding_cbow/MatMul/ReadVariableOp(w2v_embedding_cbow/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
P__inference_w2v_embedding_cbow_layer_call_and_return_conditional_losses_11958563

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddl
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
F__inference_model_12_layer_call_and_return_conditional_losses_11958754

inputsE
1w2v_embedding_cbow_matmul_readvariableop_resource:
??A
2w2v_embedding_cbow_biasadd_readvariableop_resource:	?;
'dense_12_matmul_readvariableop_resource:
??7
(dense_12_biasadd_readvariableop_resource:	?
identity??dense_12/BiasAdd/ReadVariableOp?dense_12/MatMul/ReadVariableOp?)w2v_embedding_cbow/BiasAdd/ReadVariableOp?(w2v_embedding_cbow/MatMul/ReadVariableOp?
(w2v_embedding_cbow/MatMul/ReadVariableOpReadVariableOp1w2v_embedding_cbow_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02*
(w2v_embedding_cbow/MatMul/ReadVariableOp?
w2v_embedding_cbow/MatMulMatMulinputs0w2v_embedding_cbow/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
w2v_embedding_cbow/MatMul?
)w2v_embedding_cbow/BiasAdd/ReadVariableOpReadVariableOp2w2v_embedding_cbow_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)w2v_embedding_cbow/BiasAdd/ReadVariableOp?
w2v_embedding_cbow/BiasAddBiasAdd#w2v_embedding_cbow/MatMul:product:01w2v_embedding_cbow/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
w2v_embedding_cbow/BiasAddo
average_6/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
average_6/truediv/y?
average_6/truedivRealDiv#w2v_embedding_cbow/BiasAdd:output:0average_6/truediv/y:output:0*
T0*(
_output_shapes
:??????????2
average_6/truediv?
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_12/MatMul/ReadVariableOp?
dense_12/MatMulMatMulaverage_6/truediv:z:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_12/MatMul?
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_12/BiasAdd/ReadVariableOp?
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_12/BiasAdd}
dense_12/SoftmaxSoftmaxdense_12/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_12/Softmaxv
IdentityIdentitydense_12/Softmax:softmax:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp*^w2v_embedding_cbow/BiasAdd/ReadVariableOp)^w2v_embedding_cbow/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2V
)w2v_embedding_cbow/BiasAdd/ReadVariableOp)w2v_embedding_cbow/BiasAdd/ReadVariableOp2T
(w2v_embedding_cbow/MatMul/ReadVariableOp(w2v_embedding_cbow/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
F__inference_dense_12_layer_call_and_return_conditional_losses_11958588

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddb
SoftmaxSoftmaxBiasAdd:output:0*
T0*(
_output_shapes
:??????????2	
Softmaxm
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
+__inference_model_12_layer_call_fn_11958606
input_13
unknown:
??
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_13unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_model_12_layer_call_and_return_conditional_losses_119585952
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:??????????
"
_user_specified_name
input_13
?
?
+__inference_dense_12_layer_call_fn_11958849

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_12_layer_call_and_return_conditional_losses_119585882
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
F__inference_model_12_layer_call_and_return_conditional_losses_11958701
input_13/
w2v_embedding_cbow_11958689:
??*
w2v_embedding_cbow_11958691:	?%
dense_12_11958695:
?? 
dense_12_11958697:	?
identity?? dense_12/StatefulPartitionedCall?*w2v_embedding_cbow/StatefulPartitionedCall?
*w2v_embedding_cbow/StatefulPartitionedCallStatefulPartitionedCallinput_13w2v_embedding_cbow_11958689w2v_embedding_cbow_11958691*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_w2v_embedding_cbow_layer_call_and_return_conditional_losses_119585632,
*w2v_embedding_cbow/StatefulPartitionedCall?
average_6/PartitionedCallPartitionedCall3w2v_embedding_cbow/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_average_6_layer_call_and_return_conditional_losses_119585752
average_6/PartitionedCall?
 dense_12/StatefulPartitionedCallStatefulPartitionedCall"average_6/PartitionedCall:output:0dense_12_11958695dense_12_11958697*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_12_layer_call_and_return_conditional_losses_119585882"
 dense_12/StatefulPartitionedCall?
IdentityIdentity)dense_12/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp!^dense_12/StatefulPartitionedCall+^w2v_embedding_cbow/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2X
*w2v_embedding_cbow/StatefulPartitionedCall*w2v_embedding_cbow/StatefulPartitionedCall:R N
(
_output_shapes
:??????????
"
_user_specified_name
input_13
?
?
+__inference_model_12_layer_call_fn_11958786

inputs
unknown:
??
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_model_12_layer_call_and_return_conditional_losses_119585952
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
+__inference_model_12_layer_call_fn_11958799

inputs
unknown:
??
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_model_12_layer_call_and_return_conditional_losses_119586622
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
+__inference_model_12_layer_call_fn_11958686
input_13
unknown:
??
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_13unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_model_12_layer_call_and_return_conditional_losses_119586622
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:??????????
"
_user_specified_name
input_13
?
?
&__inference_signature_wrapper_11958735
input_13
unknown:
??
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_13unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *,
f'R%
#__inference__wrapped_model_119585462
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:??????????
"
_user_specified_name
input_13
?
?
5__inference_w2v_embedding_cbow_layer_call_fn_11958818

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_w2v_embedding_cbow_layer_call_and_return_conditional_losses_119585632
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
F__inference_model_12_layer_call_and_return_conditional_losses_11958595

inputs/
w2v_embedding_cbow_11958564:
??*
w2v_embedding_cbow_11958566:	?%
dense_12_11958589:
?? 
dense_12_11958591:	?
identity?? dense_12/StatefulPartitionedCall?*w2v_embedding_cbow/StatefulPartitionedCall?
*w2v_embedding_cbow/StatefulPartitionedCallStatefulPartitionedCallinputsw2v_embedding_cbow_11958564w2v_embedding_cbow_11958566*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *Y
fTRR
P__inference_w2v_embedding_cbow_layer_call_and_return_conditional_losses_119585632,
*w2v_embedding_cbow/StatefulPartitionedCall?
average_6/PartitionedCallPartitionedCall3w2v_embedding_cbow/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_average_6_layer_call_and_return_conditional_losses_119585752
average_6/PartitionedCall?
 dense_12/StatefulPartitionedCallStatefulPartitionedCall"average_6/PartitionedCall:output:0dense_12_11958589dense_12_11958591*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dense_12_layer_call_and_return_conditional_losses_119585882"
 dense_12/StatefulPartitionedCall?
IdentityIdentity)dense_12/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp!^dense_12/StatefulPartitionedCall+^w2v_embedding_cbow/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2X
*w2v_embedding_cbow/StatefulPartitionedCall*w2v_embedding_cbow/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
P__inference_w2v_embedding_cbow_layer_call_and_return_conditional_losses_11958809

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddl
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
H
,__inference_average_6_layer_call_fn_11958829

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
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_average_6_layer_call_and_return_conditional_losses_119585752
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
c
G__inference_average_6_layer_call_and_return_conditional_losses_11958824

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
:??????????2	
truediv`
IdentityIdentitytruediv:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
>
input_132
serving_default_input_13:0??????????=
dense_121
StatefulPartitionedCall:0??????????tensorflow/serving/predict:?J
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
	optimizer
	variables
trainable_variables
regularization_losses
		keras_api


signatures
*>&call_and_return_all_conditional_losses
?__call__
@_default_save_signature"
_tf_keras_network
"
_tf_keras_input_layer
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
*A&call_and_return_all_conditional_losses
B__call__"
_tf_keras_layer
?
	variables
trainable_variables
regularization_losses
	keras_api
*C&call_and_return_all_conditional_losses
D__call__"
_tf_keras_layer
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
*E&call_and_return_all_conditional_losses
F__call__"
_tf_keras_layer
I
iter
	decay
learning_rate
momentum"
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

layers
 layer_metrics
	variables
trainable_variables
regularization_losses
!non_trainable_variables
"metrics
#layer_regularization_losses
?__call__
@_default_save_signature
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
,
Gserving_default"
signature_map
-:+
??2w2v_embedding_cbow/kernel
&:$?2w2v_embedding_cbow/bias
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

$layers
%layer_metrics
	variables
trainable_variables
regularization_losses
&non_trainable_variables
'metrics
(layer_regularization_losses
B__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

)layers
*layer_metrics
	variables
trainable_variables
regularization_losses
+non_trainable_variables
,metrics
-layer_regularization_losses
D__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses"
_generic_user_object
#:!
??2dense_12/kernel
:?2dense_12/bias
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

.layers
/layer_metrics
	variables
trainable_variables
regularization_losses
0non_trainable_variables
1metrics
2layer_regularization_losses
F__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
:	 (2SGD/iter
: (2	SGD/decay
: (2SGD/learning_rate
: (2SGD/momentum
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
30
41"
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
 "
trackable_list_wrapper
N
	5total
	6count
7	variables
8	keras_api"
_tf_keras_metric
^
	9total
	:count
;
_fn_kwargs
<	variables
=	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
50
61"
trackable_list_wrapper
-
7	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
90
:1"
trackable_list_wrapper
-
<	variables"
_generic_user_object
?2?
F__inference_model_12_layer_call_and_return_conditional_losses_11958754
F__inference_model_12_layer_call_and_return_conditional_losses_11958773
F__inference_model_12_layer_call_and_return_conditional_losses_11958701
F__inference_model_12_layer_call_and_return_conditional_losses_11958716?
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
?2?
+__inference_model_12_layer_call_fn_11958606
+__inference_model_12_layer_call_fn_11958786
+__inference_model_12_layer_call_fn_11958799
+__inference_model_12_layer_call_fn_11958686?
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
?B?
#__inference__wrapped_model_11958546input_13"?
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
?2?
P__inference_w2v_embedding_cbow_layer_call_and_return_conditional_losses_11958809?
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
5__inference_w2v_embedding_cbow_layer_call_fn_11958818?
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
G__inference_average_6_layer_call_and_return_conditional_losses_11958824?
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
,__inference_average_6_layer_call_fn_11958829?
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
F__inference_dense_12_layer_call_and_return_conditional_losses_11958840?
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
+__inference_dense_12_layer_call_fn_11958849?
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
&__inference_signature_wrapper_11958735input_13"?
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
#__inference__wrapped_model_11958546p2?/
(?%
#? 
input_13??????????
? "4?1
/
dense_12#? 
dense_12???????????
G__inference_average_6_layer_call_and_return_conditional_losses_11958824Z0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? }
,__inference_average_6_layer_call_fn_11958829M0?-
&?#
!?
inputs??????????
? "????????????
F__inference_dense_12_layer_call_and_return_conditional_losses_11958840^0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
+__inference_dense_12_layer_call_fn_11958849Q0?-
&?#
!?
inputs??????????
? "????????????
F__inference_model_12_layer_call_and_return_conditional_losses_11958701j:?7
0?-
#? 
input_13??????????
p 

 
? "&?#
?
0??????????
? ?
F__inference_model_12_layer_call_and_return_conditional_losses_11958716j:?7
0?-
#? 
input_13??????????
p

 
? "&?#
?
0??????????
? ?
F__inference_model_12_layer_call_and_return_conditional_losses_11958754h8?5
.?+
!?
inputs??????????
p 

 
? "&?#
?
0??????????
? ?
F__inference_model_12_layer_call_and_return_conditional_losses_11958773h8?5
.?+
!?
inputs??????????
p

 
? "&?#
?
0??????????
? ?
+__inference_model_12_layer_call_fn_11958606]:?7
0?-
#? 
input_13??????????
p 

 
? "????????????
+__inference_model_12_layer_call_fn_11958686]:?7
0?-
#? 
input_13??????????
p

 
? "????????????
+__inference_model_12_layer_call_fn_11958786[8?5
.?+
!?
inputs??????????
p 

 
? "????????????
+__inference_model_12_layer_call_fn_11958799[8?5
.?+
!?
inputs??????????
p

 
? "????????????
&__inference_signature_wrapper_11958735|>?;
? 
4?1
/
input_13#? 
input_13??????????"4?1
/
dense_12#? 
dense_12???????????
P__inference_w2v_embedding_cbow_layer_call_and_return_conditional_losses_11958809^0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
5__inference_w2v_embedding_cbow_layer_call_fn_11958818Q0?-
&?#
!?
inputs??????????
? "???????????