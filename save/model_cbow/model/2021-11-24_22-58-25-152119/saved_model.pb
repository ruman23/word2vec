ó
Õ¹
B
AssignVariableOp
resource
value"dtype"
dtypetype
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
delete_old_dirsbool(
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
dtypetype
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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
¾
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
executor_typestring 
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

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.6.02unknown8

w2v_embedding_cbow/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Ñ¬**
shared_namew2v_embedding_cbow/kernel

-w2v_embedding_cbow/kernel/Read/ReadVariableOpReadVariableOpw2v_embedding_cbow/kernel* 
_output_shapes
:
Ñ¬*
dtype0

w2v_embedding_cbow/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:¬*(
shared_namew2v_embedding_cbow/bias

+w2v_embedding_cbow/bias/Read/ReadVariableOpReadVariableOpw2v_embedding_cbow/bias*
_output_shapes	
:¬*
dtype0
v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
¬Ñ*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
¬Ñ*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ñ*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:Ñ*
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
õ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*°
value¦B£ B
Ù
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
	optimizer
regularization_losses
trainable_variables
	variables
		keras_api


signatures
 
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
6
iter
	decay
learning_rate
momentum
 

0
1
2
3

0
1
2
3
­

layers
regularization_losses
trainable_variables
 layer_regularization_losses
!layer_metrics
"metrics
#non_trainable_variables
	variables
 
ec
VARIABLE_VALUEw2v_embedding_cbow/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEw2v_embedding_cbow/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­

$layers
regularization_losses
trainable_variables
%layer_regularization_losses
&layer_metrics
'metrics
(non_trainable_variables
	variables
 
 
 
­

)layers
regularization_losses
trainable_variables
*layer_regularization_losses
+layer_metrics
,metrics
-non_trainable_variables
	variables
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­

.layers
regularization_losses
trainable_variables
/layer_regularization_losses
0layer_metrics
1metrics
2non_trainable_variables
	variables
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
|
serving_default_input_1Placeholder*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿÑ

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1w2v_embedding_cbow/kernelw2v_embedding_cbow/biasdense/kernel
dense/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *,
f'R%
#__inference_signature_wrapper_49935
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
¿
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename-w2v_embedding_cbow/kernel/Read/ReadVariableOp+w2v_embedding_cbow/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOpSGD/iter/Read/ReadVariableOpSGD/decay/Read/ReadVariableOp%SGD/learning_rate/Read/ReadVariableOp SGD/momentum/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOpConst*
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
GPU2 *0J 8 *'
f"R 
__inference__traced_save_50108
Ê
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamew2v_embedding_cbow/kernelw2v_embedding_cbow/biasdense/kernel
dense/biasSGD/iter	SGD/decaySGD/learning_rateSGD/momentumtotalcounttotal_1count_1*
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
GPU2 *0J 8 **
f%R#
!__inference__traced_restore_50154Üá
Á
C
'__inference_average_layer_call_fn_50023

inputs
identityÆ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_average_layer_call_and_return_conditional_losses_497752
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¬:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs
Ã
À
@__inference_model_layer_call_and_return_conditional_losses_49795

inputs,
w2v_embedding_cbow_49764:
Ñ¬'
w2v_embedding_cbow_49766:	¬
dense_49789:
¬Ñ
dense_49791:	Ñ
identity¢dense/StatefulPartitionedCall¢*w2v_embedding_cbow/StatefulPartitionedCallÉ
*w2v_embedding_cbow/StatefulPartitionedCallStatefulPartitionedCallinputsw2v_embedding_cbow_49764w2v_embedding_cbow_49766*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *V
fQRO
M__inference_w2v_embedding_cbow_layer_call_and_return_conditional_losses_497632,
*w2v_embedding_cbow/StatefulPartitionedCall
average/PartitionedCallPartitionedCall3w2v_embedding_cbow/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_average_layer_call_and_return_conditional_losses_497752
average/PartitionedCall¢
dense/StatefulPartitionedCallStatefulPartitionedCall average/PartitionedCall:output:0dense_49789dense_49791*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_497882
dense/StatefulPartitionedCall
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ2

Identity
NoOpNoOp^dense/StatefulPartitionedCall+^w2v_embedding_cbow/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÑ: : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2X
*w2v_embedding_cbow/StatefulPartitionedCall*w2v_embedding_cbow/StatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ
 
_user_specified_nameinputs

þ
 __inference__wrapped_model_49746
input_1K
7model_w2v_embedding_cbow_matmul_readvariableop_resource:
Ñ¬G
8model_w2v_embedding_cbow_biasadd_readvariableop_resource:	¬>
*model_dense_matmul_readvariableop_resource:
¬Ñ:
+model_dense_biasadd_readvariableop_resource:	Ñ
identity¢"model/dense/BiasAdd/ReadVariableOp¢!model/dense/MatMul/ReadVariableOp¢/model/w2v_embedding_cbow/BiasAdd/ReadVariableOp¢.model/w2v_embedding_cbow/MatMul/ReadVariableOpÚ
.model/w2v_embedding_cbow/MatMul/ReadVariableOpReadVariableOp7model_w2v_embedding_cbow_matmul_readvariableop_resource* 
_output_shapes
:
Ñ¬*
dtype020
.model/w2v_embedding_cbow/MatMul/ReadVariableOpÀ
model/w2v_embedding_cbow/MatMulMatMulinput_16model/w2v_embedding_cbow/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2!
model/w2v_embedding_cbow/MatMulØ
/model/w2v_embedding_cbow/BiasAdd/ReadVariableOpReadVariableOp8model_w2v_embedding_cbow_biasadd_readvariableop_resource*
_output_shapes	
:¬*
dtype021
/model/w2v_embedding_cbow/BiasAdd/ReadVariableOpæ
 model/w2v_embedding_cbow/BiasAddBiasAdd)model/w2v_embedding_cbow/MatMul:product:07model/w2v_embedding_cbow/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2"
 model/w2v_embedding_cbow/BiasAddw
model/average/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A2
model/average/truediv/y¹
model/average/truedivRealDiv)model/w2v_embedding_cbow/BiasAdd:output:0 model/average/truediv/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
model/average/truediv³
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource* 
_output_shapes
:
¬Ñ*
dtype02#
!model/dense/MatMul/ReadVariableOp«
model/dense/MatMulMatMulmodel/average/truediv:z:0)model/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ2
model/dense/MatMul±
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes	
:Ñ*
dtype02$
"model/dense/BiasAdd/ReadVariableOp²
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ2
model/dense/BiasAdd
model/dense/SoftmaxSoftmaxmodel/dense/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ2
model/dense/Softmaxy
IdentityIdentitymodel/dense/Softmax:softmax:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ2

Identityú
NoOpNoOp#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp0^model/w2v_embedding_cbow/BiasAdd/ReadVariableOp/^model/w2v_embedding_cbow/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÑ: : : : 2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2b
/model/w2v_embedding_cbow/BiasAdd/ReadVariableOp/model/w2v_embedding_cbow/BiasAdd/ReadVariableOp2`
.model/w2v_embedding_cbow/MatMul/ReadVariableOp.model/w2v_embedding_cbow/MatMul/ReadVariableOp:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ
!
_user_specified_name	input_1
Æ
Á
@__inference_model_layer_call_and_return_conditional_losses_49901
input_1,
w2v_embedding_cbow_49889:
Ñ¬'
w2v_embedding_cbow_49891:	¬
dense_49895:
¬Ñ
dense_49897:	Ñ
identity¢dense/StatefulPartitionedCall¢*w2v_embedding_cbow/StatefulPartitionedCallÊ
*w2v_embedding_cbow/StatefulPartitionedCallStatefulPartitionedCallinput_1w2v_embedding_cbow_49889w2v_embedding_cbow_49891*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *V
fQRO
M__inference_w2v_embedding_cbow_layer_call_and_return_conditional_losses_497632,
*w2v_embedding_cbow/StatefulPartitionedCall
average/PartitionedCallPartitionedCall3w2v_embedding_cbow/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_average_layer_call_and_return_conditional_losses_497752
average/PartitionedCall¢
dense/StatefulPartitionedCallStatefulPartitionedCall average/PartitionedCall:output:0dense_49895dense_49897*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_497882
dense/StatefulPartitionedCall
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ2

Identity
NoOpNoOp^dense/StatefulPartitionedCall+^w2v_embedding_cbow/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÑ: : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2X
*w2v_embedding_cbow/StatefulPartitionedCall*w2v_embedding_cbow/StatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ
!
_user_specified_name	input_1
Ì
Î
%__inference_model_layer_call_fn_49961

inputs
unknown:
Ñ¬
	unknown_0:	¬
	unknown_1:
¬Ñ
	unknown_2:	Ñ
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_498622
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÑ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ
 
_user_specified_nameinputs
Æ
Á
@__inference_model_layer_call_and_return_conditional_losses_49916
input_1,
w2v_embedding_cbow_49904:
Ñ¬'
w2v_embedding_cbow_49906:	¬
dense_49910:
¬Ñ
dense_49912:	Ñ
identity¢dense/StatefulPartitionedCall¢*w2v_embedding_cbow/StatefulPartitionedCallÊ
*w2v_embedding_cbow/StatefulPartitionedCallStatefulPartitionedCallinput_1w2v_embedding_cbow_49904w2v_embedding_cbow_49906*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *V
fQRO
M__inference_w2v_embedding_cbow_layer_call_and_return_conditional_losses_497632,
*w2v_embedding_cbow/StatefulPartitionedCall
average/PartitionedCallPartitionedCall3w2v_embedding_cbow/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_average_layer_call_and_return_conditional_losses_497752
average/PartitionedCall¢
dense/StatefulPartitionedCallStatefulPartitionedCall average/PartitionedCall:output:0dense_49910dense_49912*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_497882
dense/StatefulPartitionedCall
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ2

Identity
NoOpNoOp^dense/StatefulPartitionedCall+^w2v_embedding_cbow/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÑ: : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2X
*w2v_embedding_cbow/StatefulPartitionedCall*w2v_embedding_cbow/StatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ
!
_user_specified_name	input_1
º


M__inference_w2v_embedding_cbow_layer_call_and_return_conditional_losses_50018

inputs2
matmul_readvariableop_resource:
Ñ¬.
biasadd_readvariableop_resource:	¬
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
Ñ¬*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:¬*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2	
BiasAddl
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿÑ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ
 
_user_specified_nameinputs
º


M__inference_w2v_embedding_cbow_layer_call_and_return_conditional_losses_49763

inputs2
matmul_readvariableop_resource:
Ñ¬.
biasadd_readvariableop_resource:	¬
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
Ñ¬*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:¬*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2	
BiasAddl
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿÑ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ
 
_user_specified_nameinputs
ý#

__inference__traced_save_50108
file_prefix8
4savev2_w2v_embedding_cbow_kernel_read_readvariableop6
2savev2_w2v_embedding_cbow_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop'
#savev2_sgd_iter_read_readvariableop	(
$savev2_sgd_decay_read_readvariableop0
,savev2_sgd_learning_rate_read_readvariableop+
'savev2_sgd_momentum_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
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
Const_1
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
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*¡
valueBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names¢
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B 2
SaveV2/shape_and_slices®
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:04savev2_w2v_embedding_cbow_kernel_read_readvariableop2savev2_w2v_embedding_cbow_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop#savev2_sgd_iter_read_readvariableop$savev2_sgd_decay_read_readvariableop,savev2_sgd_learning_rate_read_readvariableop'savev2_sgd_momentum_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
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
Ñ¬:¬:
¬Ñ:Ñ: : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
Ñ¬:!

_output_shapes	
:¬:&"
 
_output_shapes
:
¬Ñ:!

_output_shapes	
:Ñ:
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
Ì
Î
%__inference_model_layer_call_fn_49948

inputs
unknown:
Ñ¬
	unknown_0:	¬
	unknown_1:
¬Ñ
	unknown_2:	Ñ
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_497952
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÑ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ
 
_user_specified_nameinputs

¢
2__inference_w2v_embedding_cbow_layer_call_fn_50008

inputs
unknown:
Ñ¬
	unknown_0:	¬
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *V
fQRO
M__inference_w2v_embedding_cbow_layer_call_and_return_conditional_losses_497632
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿÑ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ
 
_user_specified_nameinputs

ô
@__inference_dense_layer_call_and_return_conditional_losses_50049

inputs2
matmul_readvariableop_resource:
¬Ñ.
biasadd_readvariableop_resource:	Ñ
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
¬Ñ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ñ*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ2	
BiasAddb
SoftmaxSoftmaxBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ2	
Softmaxm
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¬: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs
Ñ
^
B__inference_average_layer_call_and_return_conditional_losses_50029

inputs
identity[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A2
	truediv/yl
truedivRealDivinputstruediv/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2	
truediv`
IdentityIdentitytruediv:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¬:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs
Ó
í
@__inference_model_layer_call_and_return_conditional_losses_49999

inputsE
1w2v_embedding_cbow_matmul_readvariableop_resource:
Ñ¬A
2w2v_embedding_cbow_biasadd_readvariableop_resource:	¬8
$dense_matmul_readvariableop_resource:
¬Ñ4
%dense_biasadd_readvariableop_resource:	Ñ
identity¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢)w2v_embedding_cbow/BiasAdd/ReadVariableOp¢(w2v_embedding_cbow/MatMul/ReadVariableOpÈ
(w2v_embedding_cbow/MatMul/ReadVariableOpReadVariableOp1w2v_embedding_cbow_matmul_readvariableop_resource* 
_output_shapes
:
Ñ¬*
dtype02*
(w2v_embedding_cbow/MatMul/ReadVariableOp­
w2v_embedding_cbow/MatMulMatMulinputs0w2v_embedding_cbow/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
w2v_embedding_cbow/MatMulÆ
)w2v_embedding_cbow/BiasAdd/ReadVariableOpReadVariableOp2w2v_embedding_cbow_biasadd_readvariableop_resource*
_output_shapes	
:¬*
dtype02+
)w2v_embedding_cbow/BiasAdd/ReadVariableOpÎ
w2v_embedding_cbow/BiasAddBiasAdd#w2v_embedding_cbow/MatMul:product:01w2v_embedding_cbow/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
w2v_embedding_cbow/BiasAddk
average/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A2
average/truediv/y¡
average/truedivRealDiv#w2v_embedding_cbow/BiasAdd:output:0average/truediv/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
average/truediv¡
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
¬Ñ*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMulaverage/truediv:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:Ñ*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ2
dense/BiasAddt
dense/SoftmaxSoftmaxdense/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ2
dense/Softmaxs
IdentityIdentitydense/Softmax:softmax:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ2

Identityâ
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*^w2v_embedding_cbow/BiasAdd/ReadVariableOp)^w2v_embedding_cbow/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÑ: : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2V
)w2v_embedding_cbow/BiasAdd/ReadVariableOp)w2v_embedding_cbow/BiasAdd/ReadVariableOp2T
(w2v_embedding_cbow/MatMul/ReadVariableOp(w2v_embedding_cbow/MatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ
 
_user_specified_nameinputs
Ó
í
@__inference_model_layer_call_and_return_conditional_losses_49980

inputsE
1w2v_embedding_cbow_matmul_readvariableop_resource:
Ñ¬A
2w2v_embedding_cbow_biasadd_readvariableop_resource:	¬8
$dense_matmul_readvariableop_resource:
¬Ñ4
%dense_biasadd_readvariableop_resource:	Ñ
identity¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢)w2v_embedding_cbow/BiasAdd/ReadVariableOp¢(w2v_embedding_cbow/MatMul/ReadVariableOpÈ
(w2v_embedding_cbow/MatMul/ReadVariableOpReadVariableOp1w2v_embedding_cbow_matmul_readvariableop_resource* 
_output_shapes
:
Ñ¬*
dtype02*
(w2v_embedding_cbow/MatMul/ReadVariableOp­
w2v_embedding_cbow/MatMulMatMulinputs0w2v_embedding_cbow/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
w2v_embedding_cbow/MatMulÆ
)w2v_embedding_cbow/BiasAdd/ReadVariableOpReadVariableOp2w2v_embedding_cbow_biasadd_readvariableop_resource*
_output_shapes	
:¬*
dtype02+
)w2v_embedding_cbow/BiasAdd/ReadVariableOpÎ
w2v_embedding_cbow/BiasAddBiasAdd#w2v_embedding_cbow/MatMul:product:01w2v_embedding_cbow/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
w2v_embedding_cbow/BiasAddk
average/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A2
average/truediv/y¡
average/truedivRealDiv#w2v_embedding_cbow/BiasAdd:output:0average/truediv/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
average/truediv¡
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
¬Ñ*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMulaverage/truediv:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:Ñ*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ2
dense/BiasAddt
dense/SoftmaxSoftmaxdense/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ2
dense/Softmaxs
IdentityIdentitydense/Softmax:softmax:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ2

Identityâ
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*^w2v_embedding_cbow/BiasAdd/ReadVariableOp)^w2v_embedding_cbow/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÑ: : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2V
)w2v_embedding_cbow/BiasAdd/ReadVariableOp)w2v_embedding_cbow/BiasAdd/ReadVariableOp2T
(w2v_embedding_cbow/MatMul/ReadVariableOp(w2v_embedding_cbow/MatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ
 
_user_specified_nameinputs
­
Í
#__inference_signature_wrapper_49935
input_1
unknown:
Ñ¬
	unknown_0:	¬
	unknown_1:
¬Ñ
	unknown_2:	Ñ
identity¢StatefulPartitionedCallñ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *)
f$R"
 __inference__wrapped_model_497462
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÑ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ
!
_user_specified_name	input_1
Ï
Ï
%__inference_model_layer_call_fn_49806
input_1
unknown:
Ñ¬
	unknown_0:	¬
	unknown_1:
¬Ñ
	unknown_2:	Ñ
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_497952
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÑ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ
!
_user_specified_name	input_1
ó5
â
!__inference__traced_restore_50154
file_prefix>
*assignvariableop_w2v_embedding_cbow_kernel:
Ñ¬9
*assignvariableop_1_w2v_embedding_cbow_bias:	¬3
assignvariableop_2_dense_kernel:
¬Ñ,
assignvariableop_3_dense_bias:	Ñ%
assignvariableop_4_sgd_iter:	 &
assignvariableop_5_sgd_decay: .
$assignvariableop_6_sgd_learning_rate: )
assignvariableop_7_sgd_momentum: "
assignvariableop_8_total: "
assignvariableop_9_count: %
assignvariableop_10_total_1: %
assignvariableop_11_count_1: 
identity_13¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*¡
valueBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names¨
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesì
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

Identity©
AssignVariableOpAssignVariableOp*assignvariableop_w2v_embedding_cbow_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¯
AssignVariableOp_1AssignVariableOp*assignvariableop_1_w2v_embedding_cbow_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2¤
AssignVariableOp_2AssignVariableOpassignvariableop_2_dense_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¢
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_4 
AssignVariableOp_4AssignVariableOpassignvariableop_4_sgd_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¡
AssignVariableOp_5AssignVariableOpassignvariableop_5_sgd_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6©
AssignVariableOp_6AssignVariableOp$assignvariableop_6_sgd_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7¤
AssignVariableOp_7AssignVariableOpassignvariableop_7_sgd_momentumIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8
AssignVariableOp_8AssignVariableOpassignvariableop_8_totalIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9
AssignVariableOp_9AssignVariableOpassignvariableop_9_countIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10£
AssignVariableOp_10AssignVariableOpassignvariableop_10_total_1Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11£
AssignVariableOp_11AssignVariableOpassignvariableop_11_count_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_119
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpæ
Identity_12Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_12f
Identity_13IdentityIdentity_12:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_13Î
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
Ï
Ï
%__inference_model_layer_call_fn_49886
input_1
unknown:
Ñ¬
	unknown_0:	¬
	unknown_1:
¬Ñ
	unknown_2:	Ñ
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_498622
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÑ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ
!
_user_specified_name	input_1
Ñ
^
B__inference_average_layer_call_and_return_conditional_losses_49775

inputs
identity[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `A2
	truediv/yl
truedivRealDivinputstruediv/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2	
truediv`
IdentityIdentitytruediv:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¬:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs
Ã
À
@__inference_model_layer_call_and_return_conditional_losses_49862

inputs,
w2v_embedding_cbow_49850:
Ñ¬'
w2v_embedding_cbow_49852:	¬
dense_49856:
¬Ñ
dense_49858:	Ñ
identity¢dense/StatefulPartitionedCall¢*w2v_embedding_cbow/StatefulPartitionedCallÉ
*w2v_embedding_cbow/StatefulPartitionedCallStatefulPartitionedCallinputsw2v_embedding_cbow_49850w2v_embedding_cbow_49852*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *V
fQRO
M__inference_w2v_embedding_cbow_layer_call_and_return_conditional_losses_497632,
*w2v_embedding_cbow/StatefulPartitionedCall
average/PartitionedCallPartitionedCall3w2v_embedding_cbow/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_average_layer_call_and_return_conditional_losses_497752
average/PartitionedCall¢
dense/StatefulPartitionedCallStatefulPartitionedCall average/PartitionedCall:output:0dense_49856dense_49858*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_497882
dense/StatefulPartitionedCall
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ2

Identity
NoOpNoOp^dense/StatefulPartitionedCall+^w2v_embedding_cbow/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÑ: : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2X
*w2v_embedding_cbow/StatefulPartitionedCall*w2v_embedding_cbow/StatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ
 
_user_specified_nameinputs

ô
@__inference_dense_layer_call_and_return_conditional_losses_49788

inputs2
matmul_readvariableop_resource:
¬Ñ.
biasadd_readvariableop_resource:	Ñ
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
¬Ñ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ñ*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ2	
BiasAddb
SoftmaxSoftmaxBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ2	
Softmaxm
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¬: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs
õ

%__inference_dense_layer_call_fn_50038

inputs
unknown:
¬Ñ
	unknown_0:	Ñ
identity¢StatefulPartitionedCallö
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_497882
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¬: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs"¨L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ª
serving_default
<
input_11
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿÑ:
dense1
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿÑtensorflow/serving/predict:ÁH
Ë
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
	optimizer
regularization_losses
trainable_variables
	variables
		keras_api


signatures
>__call__
?_default_save_signature
*@&call_and_return_all_conditional_losses"
_tf_keras_network
"
_tf_keras_input_layer
»

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
A__call__
*B&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
regularization_losses
trainable_variables
	variables
	keras_api
C__call__
*D&call_and_return_all_conditional_losses"
_tf_keras_layer
»

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
E__call__
*F&call_and_return_all_conditional_losses"
_tf_keras_layer
I
iter
	decay
learning_rate
momentum"
	optimizer
 "
trackable_list_wrapper
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
Ê

layers
regularization_losses
trainable_variables
 layer_regularization_losses
!layer_metrics
"metrics
#non_trainable_variables
	variables
>__call__
?_default_save_signature
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses"
_generic_user_object
,
Gserving_default"
signature_map
-:+
Ñ¬2w2v_embedding_cbow/kernel
&:$¬2w2v_embedding_cbow/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­

$layers
regularization_losses
trainable_variables
%layer_regularization_losses
&layer_metrics
'metrics
(non_trainable_variables
	variables
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­

)layers
regularization_losses
trainable_variables
*layer_regularization_losses
+layer_metrics
,metrics
-non_trainable_variables
	variables
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
 :
¬Ñ2dense/kernel
:Ñ2
dense/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­

.layers
regularization_losses
trainable_variables
/layer_regularization_losses
0layer_metrics
1metrics
2non_trainable_variables
	variables
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses"
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
trackable_list_wrapper
 "
trackable_dict_wrapper
.
30
41"
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
â2ß
%__inference_model_layer_call_fn_49806
%__inference_model_layer_call_fn_49948
%__inference_model_layer_call_fn_49961
%__inference_model_layer_call_fn_49886À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ËBÈ
 __inference__wrapped_model_49746input_1"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Î2Ë
@__inference_model_layer_call_and_return_conditional_losses_49980
@__inference_model_layer_call_and_return_conditional_losses_49999
@__inference_model_layer_call_and_return_conditional_losses_49901
@__inference_model_layer_call_and_return_conditional_losses_49916À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ü2Ù
2__inference_w2v_embedding_cbow_layer_call_fn_50008¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
÷2ô
M__inference_w2v_embedding_cbow_layer_call_and_return_conditional_losses_50018¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ñ2Î
'__inference_average_layer_call_fn_50023¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_average_layer_call_and_return_conditional_losses_50029¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ï2Ì
%__inference_dense_layer_call_fn_50038¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ê2ç
@__inference_dense_layer_call_and_return_conditional_losses_50049¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÊBÇ
#__inference_signature_wrapper_49935input_1"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 __inference__wrapped_model_49746i1¢.
'¢$
"
input_1ÿÿÿÿÿÿÿÿÿÑ
ª ".ª+
)
dense 
denseÿÿÿÿÿÿÿÿÿÑ 
B__inference_average_layer_call_and_return_conditional_losses_50029Z0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ¬
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ¬
 x
'__inference_average_layer_call_fn_50023M0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ¬
ª "ÿÿÿÿÿÿÿÿÿ¬¢
@__inference_dense_layer_call_and_return_conditional_losses_50049^0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ¬
ª "&¢#

0ÿÿÿÿÿÿÿÿÿÑ
 z
%__inference_dense_layer_call_fn_50038Q0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ¬
ª "ÿÿÿÿÿÿÿÿÿÑ­
@__inference_model_layer_call_and_return_conditional_losses_49901i9¢6
/¢,
"
input_1ÿÿÿÿÿÿÿÿÿÑ
p 

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿÑ
 ­
@__inference_model_layer_call_and_return_conditional_losses_49916i9¢6
/¢,
"
input_1ÿÿÿÿÿÿÿÿÿÑ
p

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿÑ
 ¬
@__inference_model_layer_call_and_return_conditional_losses_49980h8¢5
.¢+
!
inputsÿÿÿÿÿÿÿÿÿÑ
p 

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿÑ
 ¬
@__inference_model_layer_call_and_return_conditional_losses_49999h8¢5
.¢+
!
inputsÿÿÿÿÿÿÿÿÿÑ
p

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿÑ
 
%__inference_model_layer_call_fn_49806\9¢6
/¢,
"
input_1ÿÿÿÿÿÿÿÿÿÑ
p 

 
ª "ÿÿÿÿÿÿÿÿÿÑ
%__inference_model_layer_call_fn_49886\9¢6
/¢,
"
input_1ÿÿÿÿÿÿÿÿÿÑ
p

 
ª "ÿÿÿÿÿÿÿÿÿÑ
%__inference_model_layer_call_fn_49948[8¢5
.¢+
!
inputsÿÿÿÿÿÿÿÿÿÑ
p 

 
ª "ÿÿÿÿÿÿÿÿÿÑ
%__inference_model_layer_call_fn_49961[8¢5
.¢+
!
inputsÿÿÿÿÿÿÿÿÿÑ
p

 
ª "ÿÿÿÿÿÿÿÿÿÑ
#__inference_signature_wrapper_49935t<¢9
¢ 
2ª/
-
input_1"
input_1ÿÿÿÿÿÿÿÿÿÑ".ª+
)
dense 
denseÿÿÿÿÿÿÿÿÿÑ¯
M__inference_w2v_embedding_cbow_layer_call_and_return_conditional_losses_50018^0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿÑ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ¬
 
2__inference_w2v_embedding_cbow_layer_call_fn_50008Q0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿÑ
ª "ÿÿÿÿÿÿÿÿÿ¬