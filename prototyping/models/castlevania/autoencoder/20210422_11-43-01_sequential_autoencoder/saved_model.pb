��
��
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
�
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
executor_typestring �
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape�"serve*2.1.02unknown8��

�
encoder_hidden_layer1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	� *-
shared_nameencoder_hidden_layer1/kernel
�
0encoder_hidden_layer1/kernel/Read/ReadVariableOpReadVariableOpencoder_hidden_layer1/kernel*
_output_shapes
:	� *
dtype0
�
encoder_hidden_layer1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameencoder_hidden_layer1/bias
�
.encoder_hidden_layer1/bias/Read/ReadVariableOpReadVariableOpencoder_hidden_layer1/bias*
_output_shapes
: *
dtype0
v
latent/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_namelatent/kernel
o
!latent/kernel/Read/ReadVariableOpReadVariableOplatent/kernel*
_output_shapes

: *
dtype0
n
latent/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namelatent/bias
g
latent/bias/Read/ReadVariableOpReadVariableOplatent/bias*
_output_shapes
:*
dtype0
�
decoder_hidden_layer1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *-
shared_namedecoder_hidden_layer1/kernel
�
0decoder_hidden_layer1/kernel/Read/ReadVariableOpReadVariableOpdecoder_hidden_layer1/kernel*
_output_shapes

: *
dtype0
�
decoder_hidden_layer1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namedecoder_hidden_layer1/bias
�
.decoder_hidden_layer1/bias/Read/ReadVariableOpReadVariableOpdecoder_hidden_layer1/bias*
_output_shapes
: *
dtype0
�
decoder_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 �*&
shared_namedecoder_output/kernel
�
)decoder_output/kernel/Read/ReadVariableOpReadVariableOpdecoder_output/kernel*
_output_shapes
:	 �*
dtype0

decoder_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*$
shared_namedecoder_output/bias
x
'decoder_output/bias/Read/ReadVariableOpReadVariableOpdecoder_output/bias*
_output_shapes	
:�*
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
�
(RMSprop/encoder_hidden_layer1/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	� *9
shared_name*(RMSprop/encoder_hidden_layer1/kernel/rms
�
<RMSprop/encoder_hidden_layer1/kernel/rms/Read/ReadVariableOpReadVariableOp(RMSprop/encoder_hidden_layer1/kernel/rms*
_output_shapes
:	� *
dtype0
�
&RMSprop/encoder_hidden_layer1/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&RMSprop/encoder_hidden_layer1/bias/rms
�
:RMSprop/encoder_hidden_layer1/bias/rms/Read/ReadVariableOpReadVariableOp&RMSprop/encoder_hidden_layer1/bias/rms*
_output_shapes
: *
dtype0
�
RMSprop/latent/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
: **
shared_nameRMSprop/latent/kernel/rms
�
-RMSprop/latent/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/latent/kernel/rms*
_output_shapes

: *
dtype0
�
RMSprop/latent/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameRMSprop/latent/bias/rms

+RMSprop/latent/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/latent/bias/rms*
_output_shapes
:*
dtype0
�
(RMSprop/decoder_hidden_layer1/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *9
shared_name*(RMSprop/decoder_hidden_layer1/kernel/rms
�
<RMSprop/decoder_hidden_layer1/kernel/rms/Read/ReadVariableOpReadVariableOp(RMSprop/decoder_hidden_layer1/kernel/rms*
_output_shapes

: *
dtype0
�
&RMSprop/decoder_hidden_layer1/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&RMSprop/decoder_hidden_layer1/bias/rms
�
:RMSprop/decoder_hidden_layer1/bias/rms/Read/ReadVariableOpReadVariableOp&RMSprop/decoder_hidden_layer1/bias/rms*
_output_shapes
: *
dtype0
�
!RMSprop/decoder_output/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 �*2
shared_name#!RMSprop/decoder_output/kernel/rms
�
5RMSprop/decoder_output/kernel/rms/Read/ReadVariableOpReadVariableOp!RMSprop/decoder_output/kernel/rms*
_output_shapes
:	 �*
dtype0
�
RMSprop/decoder_output/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*0
shared_name!RMSprop/decoder_output/bias/rms
�
3RMSprop/decoder_output/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/decoder_output/bias/rms*
_output_shapes	
:�*
dtype0

NoOpNoOp
�&
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�%
value�%B�% B�%
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
	optimizer
regularization_losses
	variables
	trainable_variables

	keras_api

signatures
 
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
 regularization_losses
!	variables
"trainable_variables
#	keras_api
�
$iter
	%decay
&learning_rate
'momentum
(rho	rmsI	rmsJ	rmsK	rmsL	rmsM	rmsN	rmsO	rmsP
 
8
0
1
2
3
4
5
6
7
8
0
1
2
3
4
5
6
7
�
)non_trainable_variables
*layer_regularization_losses
regularization_losses
	variables
	trainable_variables

+layers
,metrics
 
hf
VARIABLE_VALUEencoder_hidden_layer1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEencoder_hidden_layer1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�
-layer_regularization_losses
regularization_losses
	variables

.layers
trainable_variables
/non_trainable_variables
0metrics
YW
VARIABLE_VALUElatent/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElatent/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�
1layer_regularization_losses
regularization_losses
	variables

2layers
trainable_variables
3non_trainable_variables
4metrics
hf
VARIABLE_VALUEdecoder_hidden_layer1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEdecoder_hidden_layer1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�
5layer_regularization_losses
regularization_losses
	variables

6layers
trainable_variables
7non_trainable_variables
8metrics
a_
VARIABLE_VALUEdecoder_output/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEdecoder_output/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�
9layer_regularization_losses
 regularization_losses
!	variables

:layers
"trainable_variables
;non_trainable_variables
<metrics
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
 
 

0
1
2
3

=0
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
x
	>total
	?count
@
_fn_kwargs
Aregularization_losses
B	variables
Ctrainable_variables
D	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 
 

>0
?1
 
�
Elayer_regularization_losses
Aregularization_losses
B	variables

Flayers
Ctrainable_variables
Gnon_trainable_variables
Hmetrics
 
 

>0
?1
 
��
VARIABLE_VALUE(RMSprop/encoder_hidden_layer1/kernel/rmsTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE&RMSprop/encoder_hidden_layer1/bias/rmsRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUERMSprop/latent/kernel/rmsTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUERMSprop/latent/bias/rmsRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE(RMSprop/decoder_hidden_layer1/kernel/rmsTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE&RMSprop/decoder_hidden_layer1/bias/rmsRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE!RMSprop/decoder_output/kernel/rmsTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUERMSprop/decoder_output/bias/rmsRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
�
serving_default_encoder_inputPlaceholder*,
_output_shapes
:����������*
dtype0*!
shape:����������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_encoder_inputencoder_hidden_layer1/kernelencoder_hidden_layer1/biaslatent/kernellatent/biasdecoder_hidden_layer1/kerneldecoder_hidden_layer1/biasdecoder_output/kerneldecoder_output/bias*
Tin
2	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*,
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*,
f'R%
#__inference_signature_wrapper_19820
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename0encoder_hidden_layer1/kernel/Read/ReadVariableOp.encoder_hidden_layer1/bias/Read/ReadVariableOp!latent/kernel/Read/ReadVariableOplatent/bias/Read/ReadVariableOp0decoder_hidden_layer1/kernel/Read/ReadVariableOp.decoder_hidden_layer1/bias/Read/ReadVariableOp)decoder_output/kernel/Read/ReadVariableOp'decoder_output/bias/Read/ReadVariableOp RMSprop/iter/Read/ReadVariableOp!RMSprop/decay/Read/ReadVariableOp)RMSprop/learning_rate/Read/ReadVariableOp$RMSprop/momentum/Read/ReadVariableOpRMSprop/rho/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp<RMSprop/encoder_hidden_layer1/kernel/rms/Read/ReadVariableOp:RMSprop/encoder_hidden_layer1/bias/rms/Read/ReadVariableOp-RMSprop/latent/kernel/rms/Read/ReadVariableOp+RMSprop/latent/bias/rms/Read/ReadVariableOp<RMSprop/decoder_hidden_layer1/kernel/rms/Read/ReadVariableOp:RMSprop/decoder_hidden_layer1/bias/rms/Read/ReadVariableOp5RMSprop/decoder_output/kernel/rms/Read/ReadVariableOp3RMSprop/decoder_output/bias/rms/Read/ReadVariableOpConst*$
Tin
2	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: *-
config_proto

CPU

GPU2*0J 8*'
f"R 
__inference__traced_save_20363
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameencoder_hidden_layer1/kernelencoder_hidden_layer1/biaslatent/kernellatent/biasdecoder_hidden_layer1/kerneldecoder_hidden_layer1/biasdecoder_output/kerneldecoder_output/biasRMSprop/iterRMSprop/decayRMSprop/learning_rateRMSprop/momentumRMSprop/rhototalcount(RMSprop/encoder_hidden_layer1/kernel/rms&RMSprop/encoder_hidden_layer1/bias/rmsRMSprop/latent/kernel/rmsRMSprop/latent/bias/rms(RMSprop/decoder_hidden_layer1/kernel/rms&RMSprop/decoder_hidden_layer1/bias/rms!RMSprop/decoder_output/kernel/rmsRMSprop/decoder_output/bias/rms*#
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: *-
config_proto

CPU

GPU2*0J 8**
f%R#
!__inference__traced_restore_20444��	
�$
�
A__inference_latent_layer_call_and_return_conditional_losses_20179

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp�
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

: *
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis�
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis�
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const�
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1�
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis�
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat�
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:��������� 2
Tensordot/transpose�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
Tensordot/Reshape�
Tensordot/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/transpose_1/perm�
Tensordot/transpose_1	Transpose Tensordot/ReadVariableOp:value:0#Tensordot/transpose_1/perm:output:0*
T0*
_output_shapes

: 2
Tensordot/transpose_1�
Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/Reshape_1/shape�
Tensordot/Reshape_1ReshapeTensordot/transpose_1:y:0"Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes

: 2
Tensordot/Reshape_1�
Tensordot/MatMulMatMulTensordot/Reshape:output:0Tensordot/Reshape_1:output:0*
T0*'
_output_shapes
:���������2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis�
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������2
	Tensordot�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������2	
BiasAdd\
TanhTanhBiasAdd:output:0*
T0*+
_output_shapes
:���������2
Tanh�
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:���������2

Identity"
identityIdentity:output:0*2
_input_shapes!
:��������� ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:& "
 
_user_specified_nameinputs
�d
�
!__inference__traced_restore_20444
file_prefix1
-assignvariableop_encoder_hidden_layer1_kernel1
-assignvariableop_1_encoder_hidden_layer1_bias$
 assignvariableop_2_latent_kernel"
assignvariableop_3_latent_bias3
/assignvariableop_4_decoder_hidden_layer1_kernel1
-assignvariableop_5_decoder_hidden_layer1_bias,
(assignvariableop_6_decoder_output_kernel*
&assignvariableop_7_decoder_output_bias#
assignvariableop_8_rmsprop_iter$
 assignvariableop_9_rmsprop_decay-
)assignvariableop_10_rmsprop_learning_rate(
$assignvariableop_11_rmsprop_momentum#
assignvariableop_12_rmsprop_rho
assignvariableop_13_total
assignvariableop_14_count@
<assignvariableop_15_rmsprop_encoder_hidden_layer1_kernel_rms>
:assignvariableop_16_rmsprop_encoder_hidden_layer1_bias_rms1
-assignvariableop_17_rmsprop_latent_kernel_rms/
+assignvariableop_18_rmsprop_latent_bias_rms@
<assignvariableop_19_rmsprop_decoder_hidden_layer1_kernel_rms>
:assignvariableop_20_rmsprop_decoder_hidden_layer1_bias_rms9
5assignvariableop_21_rmsprop_decoder_output_kernel_rms7
3assignvariableop_22_rmsprop_decoder_output_bias_rms
identity_24��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�	RestoreV2�RestoreV2_1�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*p
_output_shapes^
\:::::::::::::::::::::::*%
dtypes
2	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp-assignvariableop_encoder_hidden_layer1_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp-assignvariableop_1_encoder_hidden_layer1_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp assignvariableop_2_latent_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOpassignvariableop_3_latent_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp/assignvariableop_4_decoder_hidden_layer1_kernelIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp-assignvariableop_5_decoder_hidden_layer1_biasIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp(assignvariableop_6_decoder_output_kernelIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp&assignvariableop_7_decoder_output_biasIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0	*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOpassignvariableop_8_rmsprop_iterIdentity_8:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp assignvariableop_9_rmsprop_decayIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp)assignvariableop_10_rmsprop_learning_rateIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp$assignvariableop_11_rmsprop_momentumIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOpassignvariableop_12_rmsprop_rhoIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp<assignvariableop_15_rmsprop_encoder_hidden_layer1_kernel_rmsIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp:assignvariableop_16_rmsprop_encoder_hidden_layer1_bias_rmsIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp-assignvariableop_17_rmsprop_latent_kernel_rmsIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp+assignvariableop_18_rmsprop_latent_bias_rmsIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp<assignvariableop_19_rmsprop_decoder_hidden_layer1_kernel_rmsIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp:assignvariableop_20_rmsprop_decoder_hidden_layer1_bias_rmsIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp5assignvariableop_21_rmsprop_decoder_output_kernel_rmsIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp3assignvariableop_22_rmsprop_decoder_output_bias_rmsIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22�
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names�
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices�
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_23Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_23�
Identity_24IdentityIdentity_23:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_24"#
identity_24Identity_24:output:0*q
_input_shapes`
^: :::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:+ '
%
_user_specified_namefile_prefix
�

�
+__inference_autoencoder_layer_call_fn_19798
encoder_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallencoder_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*
Tin
2	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*,
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_autoencoder_layer_call_and_return_conditional_losses_197872
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:����������::::::::22
StatefulPartitionedCallStatefulPartitionedCall:- )
'
_user_specified_nameencoder_input
�$
�
I__inference_decoder_output_layer_call_and_return_conditional_losses_20263

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp�
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	 �*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis�
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis�
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const�
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1�
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis�
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat�
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:��������� 2
Tensordot/transpose�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
Tensordot/Reshape�
Tensordot/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/transpose_1/perm�
Tensordot/transpose_1	Transpose Tensordot/ReadVariableOp:value:0#Tensordot/transpose_1/perm:output:0*
T0*
_output_shapes
:	 �2
Tensordot/transpose_1�
Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"    �   2
Tensordot/Reshape_1/shape�
Tensordot/Reshape_1ReshapeTensordot/transpose_1:y:0"Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes
:	 �2
Tensordot/Reshape_1�
Tensordot/MatMulMatMulTensordot/Reshape:output:0Tensordot/Reshape_1:output:0*
T0*(
_output_shapes
:����������2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis�
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:����������2
	Tensordot�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������2	
BiasAddf
SigmoidSigmoidBiasAdd:output:0*
T0*,
_output_shapes
:����������2	
Sigmoid�
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*2
_input_shapes!
:��������� ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:& "
 
_user_specified_nameinputs
��
�
 __inference__wrapped_model_19530
encoder_inputG
Cautoencoder_encoder_hidden_layer1_tensordot_readvariableop_resourceE
Aautoencoder_encoder_hidden_layer1_biasadd_readvariableop_resource8
4autoencoder_latent_tensordot_readvariableop_resource6
2autoencoder_latent_biasadd_readvariableop_resourceG
Cautoencoder_decoder_hidden_layer1_tensordot_readvariableop_resourceE
Aautoencoder_decoder_hidden_layer1_biasadd_readvariableop_resource@
<autoencoder_decoder_output_tensordot_readvariableop_resource>
:autoencoder_decoder_output_biasadd_readvariableop_resource
identity��8autoencoder/decoder_hidden_layer1/BiasAdd/ReadVariableOp�:autoencoder/decoder_hidden_layer1/Tensordot/ReadVariableOp�1autoencoder/decoder_output/BiasAdd/ReadVariableOp�3autoencoder/decoder_output/Tensordot/ReadVariableOp�8autoencoder/encoder_hidden_layer1/BiasAdd/ReadVariableOp�:autoencoder/encoder_hidden_layer1/Tensordot/ReadVariableOp�)autoencoder/latent/BiasAdd/ReadVariableOp�+autoencoder/latent/Tensordot/ReadVariableOp�
:autoencoder/encoder_hidden_layer1/Tensordot/ReadVariableOpReadVariableOpCautoencoder_encoder_hidden_layer1_tensordot_readvariableop_resource*
_output_shapes
:	� *
dtype02<
:autoencoder/encoder_hidden_layer1/Tensordot/ReadVariableOp�
0autoencoder/encoder_hidden_layer1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:22
0autoencoder/encoder_hidden_layer1/Tensordot/axes�
0autoencoder/encoder_hidden_layer1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       22
0autoencoder/encoder_hidden_layer1/Tensordot/free�
1autoencoder/encoder_hidden_layer1/Tensordot/ShapeShapeencoder_input*
T0*
_output_shapes
:23
1autoencoder/encoder_hidden_layer1/Tensordot/Shape�
9autoencoder/encoder_hidden_layer1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9autoencoder/encoder_hidden_layer1/Tensordot/GatherV2/axis�
4autoencoder/encoder_hidden_layer1/Tensordot/GatherV2GatherV2:autoencoder/encoder_hidden_layer1/Tensordot/Shape:output:09autoencoder/encoder_hidden_layer1/Tensordot/free:output:0Bautoencoder/encoder_hidden_layer1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:26
4autoencoder/encoder_hidden_layer1/Tensordot/GatherV2�
;autoencoder/encoder_hidden_layer1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2=
;autoencoder/encoder_hidden_layer1/Tensordot/GatherV2_1/axis�
6autoencoder/encoder_hidden_layer1/Tensordot/GatherV2_1GatherV2:autoencoder/encoder_hidden_layer1/Tensordot/Shape:output:09autoencoder/encoder_hidden_layer1/Tensordot/axes:output:0Dautoencoder/encoder_hidden_layer1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:28
6autoencoder/encoder_hidden_layer1/Tensordot/GatherV2_1�
1autoencoder/encoder_hidden_layer1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 23
1autoencoder/encoder_hidden_layer1/Tensordot/Const�
0autoencoder/encoder_hidden_layer1/Tensordot/ProdProd=autoencoder/encoder_hidden_layer1/Tensordot/GatherV2:output:0:autoencoder/encoder_hidden_layer1/Tensordot/Const:output:0*
T0*
_output_shapes
: 22
0autoencoder/encoder_hidden_layer1/Tensordot/Prod�
3autoencoder/encoder_hidden_layer1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 25
3autoencoder/encoder_hidden_layer1/Tensordot/Const_1�
2autoencoder/encoder_hidden_layer1/Tensordot/Prod_1Prod?autoencoder/encoder_hidden_layer1/Tensordot/GatherV2_1:output:0<autoencoder/encoder_hidden_layer1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 24
2autoencoder/encoder_hidden_layer1/Tensordot/Prod_1�
7autoencoder/encoder_hidden_layer1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7autoencoder/encoder_hidden_layer1/Tensordot/concat/axis�
2autoencoder/encoder_hidden_layer1/Tensordot/concatConcatV29autoencoder/encoder_hidden_layer1/Tensordot/free:output:09autoencoder/encoder_hidden_layer1/Tensordot/axes:output:0@autoencoder/encoder_hidden_layer1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:24
2autoencoder/encoder_hidden_layer1/Tensordot/concat�
1autoencoder/encoder_hidden_layer1/Tensordot/stackPack9autoencoder/encoder_hidden_layer1/Tensordot/Prod:output:0;autoencoder/encoder_hidden_layer1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:23
1autoencoder/encoder_hidden_layer1/Tensordot/stack�
5autoencoder/encoder_hidden_layer1/Tensordot/transpose	Transposeencoder_input;autoencoder/encoder_hidden_layer1/Tensordot/concat:output:0*
T0*,
_output_shapes
:����������27
5autoencoder/encoder_hidden_layer1/Tensordot/transpose�
3autoencoder/encoder_hidden_layer1/Tensordot/ReshapeReshape9autoencoder/encoder_hidden_layer1/Tensordot/transpose:y:0:autoencoder/encoder_hidden_layer1/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������25
3autoencoder/encoder_hidden_layer1/Tensordot/Reshape�
<autoencoder/encoder_hidden_layer1/Tensordot/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2>
<autoencoder/encoder_hidden_layer1/Tensordot/transpose_1/perm�
7autoencoder/encoder_hidden_layer1/Tensordot/transpose_1	TransposeBautoencoder/encoder_hidden_layer1/Tensordot/ReadVariableOp:value:0Eautoencoder/encoder_hidden_layer1/Tensordot/transpose_1/perm:output:0*
T0*
_output_shapes
:	� 29
7autoencoder/encoder_hidden_layer1/Tensordot/transpose_1�
;autoencoder/encoder_hidden_layer1/Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"�       2=
;autoencoder/encoder_hidden_layer1/Tensordot/Reshape_1/shape�
5autoencoder/encoder_hidden_layer1/Tensordot/Reshape_1Reshape;autoencoder/encoder_hidden_layer1/Tensordot/transpose_1:y:0Dautoencoder/encoder_hidden_layer1/Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes
:	� 27
5autoencoder/encoder_hidden_layer1/Tensordot/Reshape_1�
2autoencoder/encoder_hidden_layer1/Tensordot/MatMulMatMul<autoencoder/encoder_hidden_layer1/Tensordot/Reshape:output:0>autoencoder/encoder_hidden_layer1/Tensordot/Reshape_1:output:0*
T0*'
_output_shapes
:��������� 24
2autoencoder/encoder_hidden_layer1/Tensordot/MatMul�
3autoencoder/encoder_hidden_layer1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 25
3autoencoder/encoder_hidden_layer1/Tensordot/Const_2�
9autoencoder/encoder_hidden_layer1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9autoencoder/encoder_hidden_layer1/Tensordot/concat_1/axis�
4autoencoder/encoder_hidden_layer1/Tensordot/concat_1ConcatV2=autoencoder/encoder_hidden_layer1/Tensordot/GatherV2:output:0<autoencoder/encoder_hidden_layer1/Tensordot/Const_2:output:0Bautoencoder/encoder_hidden_layer1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:26
4autoencoder/encoder_hidden_layer1/Tensordot/concat_1�
+autoencoder/encoder_hidden_layer1/TensordotReshape<autoencoder/encoder_hidden_layer1/Tensordot/MatMul:product:0=autoencoder/encoder_hidden_layer1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:��������� 2-
+autoencoder/encoder_hidden_layer1/Tensordot�
8autoencoder/encoder_hidden_layer1/BiasAdd/ReadVariableOpReadVariableOpAautoencoder_encoder_hidden_layer1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02:
8autoencoder/encoder_hidden_layer1/BiasAdd/ReadVariableOp�
)autoencoder/encoder_hidden_layer1/BiasAddBiasAdd4autoencoder/encoder_hidden_layer1/Tensordot:output:0@autoencoder/encoder_hidden_layer1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:��������� 2+
)autoencoder/encoder_hidden_layer1/BiasAdd�
&autoencoder/encoder_hidden_layer1/TanhTanh2autoencoder/encoder_hidden_layer1/BiasAdd:output:0*
T0*+
_output_shapes
:��������� 2(
&autoencoder/encoder_hidden_layer1/Tanh�
+autoencoder/latent/Tensordot/ReadVariableOpReadVariableOp4autoencoder_latent_tensordot_readvariableop_resource*
_output_shapes

: *
dtype02-
+autoencoder/latent/Tensordot/ReadVariableOp�
!autoencoder/latent/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2#
!autoencoder/latent/Tensordot/axes�
!autoencoder/latent/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2#
!autoencoder/latent/Tensordot/free�
"autoencoder/latent/Tensordot/ShapeShape*autoencoder/encoder_hidden_layer1/Tanh:y:0*
T0*
_output_shapes
:2$
"autoencoder/latent/Tensordot/Shape�
*autoencoder/latent/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*autoencoder/latent/Tensordot/GatherV2/axis�
%autoencoder/latent/Tensordot/GatherV2GatherV2+autoencoder/latent/Tensordot/Shape:output:0*autoencoder/latent/Tensordot/free:output:03autoencoder/latent/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2'
%autoencoder/latent/Tensordot/GatherV2�
,autoencoder/latent/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,autoencoder/latent/Tensordot/GatherV2_1/axis�
'autoencoder/latent/Tensordot/GatherV2_1GatherV2+autoencoder/latent/Tensordot/Shape:output:0*autoencoder/latent/Tensordot/axes:output:05autoencoder/latent/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'autoencoder/latent/Tensordot/GatherV2_1�
"autoencoder/latent/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"autoencoder/latent/Tensordot/Const�
!autoencoder/latent/Tensordot/ProdProd.autoencoder/latent/Tensordot/GatherV2:output:0+autoencoder/latent/Tensordot/Const:output:0*
T0*
_output_shapes
: 2#
!autoencoder/latent/Tensordot/Prod�
$autoencoder/latent/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$autoencoder/latent/Tensordot/Const_1�
#autoencoder/latent/Tensordot/Prod_1Prod0autoencoder/latent/Tensordot/GatherV2_1:output:0-autoencoder/latent/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2%
#autoencoder/latent/Tensordot/Prod_1�
(autoencoder/latent/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2*
(autoencoder/latent/Tensordot/concat/axis�
#autoencoder/latent/Tensordot/concatConcatV2*autoencoder/latent/Tensordot/free:output:0*autoencoder/latent/Tensordot/axes:output:01autoencoder/latent/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2%
#autoencoder/latent/Tensordot/concat�
"autoencoder/latent/Tensordot/stackPack*autoencoder/latent/Tensordot/Prod:output:0,autoencoder/latent/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2$
"autoencoder/latent/Tensordot/stack�
&autoencoder/latent/Tensordot/transpose	Transpose*autoencoder/encoder_hidden_layer1/Tanh:y:0,autoencoder/latent/Tensordot/concat:output:0*
T0*+
_output_shapes
:��������� 2(
&autoencoder/latent/Tensordot/transpose�
$autoencoder/latent/Tensordot/ReshapeReshape*autoencoder/latent/Tensordot/transpose:y:0+autoencoder/latent/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2&
$autoencoder/latent/Tensordot/Reshape�
-autoencoder/latent/Tensordot/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2/
-autoencoder/latent/Tensordot/transpose_1/perm�
(autoencoder/latent/Tensordot/transpose_1	Transpose3autoencoder/latent/Tensordot/ReadVariableOp:value:06autoencoder/latent/Tensordot/transpose_1/perm:output:0*
T0*
_output_shapes

: 2*
(autoencoder/latent/Tensordot/transpose_1�
,autoencoder/latent/Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"       2.
,autoencoder/latent/Tensordot/Reshape_1/shape�
&autoencoder/latent/Tensordot/Reshape_1Reshape,autoencoder/latent/Tensordot/transpose_1:y:05autoencoder/latent/Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes

: 2(
&autoencoder/latent/Tensordot/Reshape_1�
#autoencoder/latent/Tensordot/MatMulMatMul-autoencoder/latent/Tensordot/Reshape:output:0/autoencoder/latent/Tensordot/Reshape_1:output:0*
T0*'
_output_shapes
:���������2%
#autoencoder/latent/Tensordot/MatMul�
$autoencoder/latent/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$autoencoder/latent/Tensordot/Const_2�
*autoencoder/latent/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*autoencoder/latent/Tensordot/concat_1/axis�
%autoencoder/latent/Tensordot/concat_1ConcatV2.autoencoder/latent/Tensordot/GatherV2:output:0-autoencoder/latent/Tensordot/Const_2:output:03autoencoder/latent/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2'
%autoencoder/latent/Tensordot/concat_1�
autoencoder/latent/TensordotReshape-autoencoder/latent/Tensordot/MatMul:product:0.autoencoder/latent/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������2
autoencoder/latent/Tensordot�
)autoencoder/latent/BiasAdd/ReadVariableOpReadVariableOp2autoencoder_latent_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)autoencoder/latent/BiasAdd/ReadVariableOp�
autoencoder/latent/BiasAddBiasAdd%autoencoder/latent/Tensordot:output:01autoencoder/latent/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������2
autoencoder/latent/BiasAdd�
autoencoder/latent/TanhTanh#autoencoder/latent/BiasAdd:output:0*
T0*+
_output_shapes
:���������2
autoencoder/latent/Tanh�
:autoencoder/decoder_hidden_layer1/Tensordot/ReadVariableOpReadVariableOpCautoencoder_decoder_hidden_layer1_tensordot_readvariableop_resource*
_output_shapes

: *
dtype02<
:autoencoder/decoder_hidden_layer1/Tensordot/ReadVariableOp�
0autoencoder/decoder_hidden_layer1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:22
0autoencoder/decoder_hidden_layer1/Tensordot/axes�
0autoencoder/decoder_hidden_layer1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       22
0autoencoder/decoder_hidden_layer1/Tensordot/free�
1autoencoder/decoder_hidden_layer1/Tensordot/ShapeShapeautoencoder/latent/Tanh:y:0*
T0*
_output_shapes
:23
1autoencoder/decoder_hidden_layer1/Tensordot/Shape�
9autoencoder/decoder_hidden_layer1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9autoencoder/decoder_hidden_layer1/Tensordot/GatherV2/axis�
4autoencoder/decoder_hidden_layer1/Tensordot/GatherV2GatherV2:autoencoder/decoder_hidden_layer1/Tensordot/Shape:output:09autoencoder/decoder_hidden_layer1/Tensordot/free:output:0Bautoencoder/decoder_hidden_layer1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:26
4autoencoder/decoder_hidden_layer1/Tensordot/GatherV2�
;autoencoder/decoder_hidden_layer1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2=
;autoencoder/decoder_hidden_layer1/Tensordot/GatherV2_1/axis�
6autoencoder/decoder_hidden_layer1/Tensordot/GatherV2_1GatherV2:autoencoder/decoder_hidden_layer1/Tensordot/Shape:output:09autoencoder/decoder_hidden_layer1/Tensordot/axes:output:0Dautoencoder/decoder_hidden_layer1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:28
6autoencoder/decoder_hidden_layer1/Tensordot/GatherV2_1�
1autoencoder/decoder_hidden_layer1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 23
1autoencoder/decoder_hidden_layer1/Tensordot/Const�
0autoencoder/decoder_hidden_layer1/Tensordot/ProdProd=autoencoder/decoder_hidden_layer1/Tensordot/GatherV2:output:0:autoencoder/decoder_hidden_layer1/Tensordot/Const:output:0*
T0*
_output_shapes
: 22
0autoencoder/decoder_hidden_layer1/Tensordot/Prod�
3autoencoder/decoder_hidden_layer1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 25
3autoencoder/decoder_hidden_layer1/Tensordot/Const_1�
2autoencoder/decoder_hidden_layer1/Tensordot/Prod_1Prod?autoencoder/decoder_hidden_layer1/Tensordot/GatherV2_1:output:0<autoencoder/decoder_hidden_layer1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 24
2autoencoder/decoder_hidden_layer1/Tensordot/Prod_1�
7autoencoder/decoder_hidden_layer1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7autoencoder/decoder_hidden_layer1/Tensordot/concat/axis�
2autoencoder/decoder_hidden_layer1/Tensordot/concatConcatV29autoencoder/decoder_hidden_layer1/Tensordot/free:output:09autoencoder/decoder_hidden_layer1/Tensordot/axes:output:0@autoencoder/decoder_hidden_layer1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:24
2autoencoder/decoder_hidden_layer1/Tensordot/concat�
1autoencoder/decoder_hidden_layer1/Tensordot/stackPack9autoencoder/decoder_hidden_layer1/Tensordot/Prod:output:0;autoencoder/decoder_hidden_layer1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:23
1autoencoder/decoder_hidden_layer1/Tensordot/stack�
5autoencoder/decoder_hidden_layer1/Tensordot/transpose	Transposeautoencoder/latent/Tanh:y:0;autoencoder/decoder_hidden_layer1/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������27
5autoencoder/decoder_hidden_layer1/Tensordot/transpose�
3autoencoder/decoder_hidden_layer1/Tensordot/ReshapeReshape9autoencoder/decoder_hidden_layer1/Tensordot/transpose:y:0:autoencoder/decoder_hidden_layer1/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������25
3autoencoder/decoder_hidden_layer1/Tensordot/Reshape�
<autoencoder/decoder_hidden_layer1/Tensordot/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2>
<autoencoder/decoder_hidden_layer1/Tensordot/transpose_1/perm�
7autoencoder/decoder_hidden_layer1/Tensordot/transpose_1	TransposeBautoencoder/decoder_hidden_layer1/Tensordot/ReadVariableOp:value:0Eautoencoder/decoder_hidden_layer1/Tensordot/transpose_1/perm:output:0*
T0*
_output_shapes

: 29
7autoencoder/decoder_hidden_layer1/Tensordot/transpose_1�
;autoencoder/decoder_hidden_layer1/Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"       2=
;autoencoder/decoder_hidden_layer1/Tensordot/Reshape_1/shape�
5autoencoder/decoder_hidden_layer1/Tensordot/Reshape_1Reshape;autoencoder/decoder_hidden_layer1/Tensordot/transpose_1:y:0Dautoencoder/decoder_hidden_layer1/Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes

: 27
5autoencoder/decoder_hidden_layer1/Tensordot/Reshape_1�
2autoencoder/decoder_hidden_layer1/Tensordot/MatMulMatMul<autoencoder/decoder_hidden_layer1/Tensordot/Reshape:output:0>autoencoder/decoder_hidden_layer1/Tensordot/Reshape_1:output:0*
T0*'
_output_shapes
:��������� 24
2autoencoder/decoder_hidden_layer1/Tensordot/MatMul�
3autoencoder/decoder_hidden_layer1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 25
3autoencoder/decoder_hidden_layer1/Tensordot/Const_2�
9autoencoder/decoder_hidden_layer1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9autoencoder/decoder_hidden_layer1/Tensordot/concat_1/axis�
4autoencoder/decoder_hidden_layer1/Tensordot/concat_1ConcatV2=autoencoder/decoder_hidden_layer1/Tensordot/GatherV2:output:0<autoencoder/decoder_hidden_layer1/Tensordot/Const_2:output:0Bautoencoder/decoder_hidden_layer1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:26
4autoencoder/decoder_hidden_layer1/Tensordot/concat_1�
+autoencoder/decoder_hidden_layer1/TensordotReshape<autoencoder/decoder_hidden_layer1/Tensordot/MatMul:product:0=autoencoder/decoder_hidden_layer1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:��������� 2-
+autoencoder/decoder_hidden_layer1/Tensordot�
8autoencoder/decoder_hidden_layer1/BiasAdd/ReadVariableOpReadVariableOpAautoencoder_decoder_hidden_layer1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02:
8autoencoder/decoder_hidden_layer1/BiasAdd/ReadVariableOp�
)autoencoder/decoder_hidden_layer1/BiasAddBiasAdd4autoencoder/decoder_hidden_layer1/Tensordot:output:0@autoencoder/decoder_hidden_layer1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:��������� 2+
)autoencoder/decoder_hidden_layer1/BiasAdd�
&autoencoder/decoder_hidden_layer1/TanhTanh2autoencoder/decoder_hidden_layer1/BiasAdd:output:0*
T0*+
_output_shapes
:��������� 2(
&autoencoder/decoder_hidden_layer1/Tanh�
3autoencoder/decoder_output/Tensordot/ReadVariableOpReadVariableOp<autoencoder_decoder_output_tensordot_readvariableop_resource*
_output_shapes
:	 �*
dtype025
3autoencoder/decoder_output/Tensordot/ReadVariableOp�
)autoencoder/decoder_output/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2+
)autoencoder/decoder_output/Tensordot/axes�
)autoencoder/decoder_output/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2+
)autoencoder/decoder_output/Tensordot/free�
*autoencoder/decoder_output/Tensordot/ShapeShape*autoencoder/decoder_hidden_layer1/Tanh:y:0*
T0*
_output_shapes
:2,
*autoencoder/decoder_output/Tensordot/Shape�
2autoencoder/decoder_output/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 24
2autoencoder/decoder_output/Tensordot/GatherV2/axis�
-autoencoder/decoder_output/Tensordot/GatherV2GatherV23autoencoder/decoder_output/Tensordot/Shape:output:02autoencoder/decoder_output/Tensordot/free:output:0;autoencoder/decoder_output/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2/
-autoencoder/decoder_output/Tensordot/GatherV2�
4autoencoder/decoder_output/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 26
4autoencoder/decoder_output/Tensordot/GatherV2_1/axis�
/autoencoder/decoder_output/Tensordot/GatherV2_1GatherV23autoencoder/decoder_output/Tensordot/Shape:output:02autoencoder/decoder_output/Tensordot/axes:output:0=autoencoder/decoder_output/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:21
/autoencoder/decoder_output/Tensordot/GatherV2_1�
*autoencoder/decoder_output/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2,
*autoencoder/decoder_output/Tensordot/Const�
)autoencoder/decoder_output/Tensordot/ProdProd6autoencoder/decoder_output/Tensordot/GatherV2:output:03autoencoder/decoder_output/Tensordot/Const:output:0*
T0*
_output_shapes
: 2+
)autoencoder/decoder_output/Tensordot/Prod�
,autoencoder/decoder_output/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,autoencoder/decoder_output/Tensordot/Const_1�
+autoencoder/decoder_output/Tensordot/Prod_1Prod8autoencoder/decoder_output/Tensordot/GatherV2_1:output:05autoencoder/decoder_output/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2-
+autoencoder/decoder_output/Tensordot/Prod_1�
0autoencoder/decoder_output/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0autoencoder/decoder_output/Tensordot/concat/axis�
+autoencoder/decoder_output/Tensordot/concatConcatV22autoencoder/decoder_output/Tensordot/free:output:02autoencoder/decoder_output/Tensordot/axes:output:09autoencoder/decoder_output/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2-
+autoencoder/decoder_output/Tensordot/concat�
*autoencoder/decoder_output/Tensordot/stackPack2autoencoder/decoder_output/Tensordot/Prod:output:04autoencoder/decoder_output/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2,
*autoencoder/decoder_output/Tensordot/stack�
.autoencoder/decoder_output/Tensordot/transpose	Transpose*autoencoder/decoder_hidden_layer1/Tanh:y:04autoencoder/decoder_output/Tensordot/concat:output:0*
T0*+
_output_shapes
:��������� 20
.autoencoder/decoder_output/Tensordot/transpose�
,autoencoder/decoder_output/Tensordot/ReshapeReshape2autoencoder/decoder_output/Tensordot/transpose:y:03autoencoder/decoder_output/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2.
,autoencoder/decoder_output/Tensordot/Reshape�
5autoencoder/decoder_output/Tensordot/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       27
5autoencoder/decoder_output/Tensordot/transpose_1/perm�
0autoencoder/decoder_output/Tensordot/transpose_1	Transpose;autoencoder/decoder_output/Tensordot/ReadVariableOp:value:0>autoencoder/decoder_output/Tensordot/transpose_1/perm:output:0*
T0*
_output_shapes
:	 �22
0autoencoder/decoder_output/Tensordot/transpose_1�
4autoencoder/decoder_output/Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"    �   26
4autoencoder/decoder_output/Tensordot/Reshape_1/shape�
.autoencoder/decoder_output/Tensordot/Reshape_1Reshape4autoencoder/decoder_output/Tensordot/transpose_1:y:0=autoencoder/decoder_output/Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes
:	 �20
.autoencoder/decoder_output/Tensordot/Reshape_1�
+autoencoder/decoder_output/Tensordot/MatMulMatMul5autoencoder/decoder_output/Tensordot/Reshape:output:07autoencoder/decoder_output/Tensordot/Reshape_1:output:0*
T0*(
_output_shapes
:����������2-
+autoencoder/decoder_output/Tensordot/MatMul�
,autoencoder/decoder_output/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�2.
,autoencoder/decoder_output/Tensordot/Const_2�
2autoencoder/decoder_output/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 24
2autoencoder/decoder_output/Tensordot/concat_1/axis�
-autoencoder/decoder_output/Tensordot/concat_1ConcatV26autoencoder/decoder_output/Tensordot/GatherV2:output:05autoencoder/decoder_output/Tensordot/Const_2:output:0;autoencoder/decoder_output/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2/
-autoencoder/decoder_output/Tensordot/concat_1�
$autoencoder/decoder_output/TensordotReshape5autoencoder/decoder_output/Tensordot/MatMul:product:06autoencoder/decoder_output/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:����������2&
$autoencoder/decoder_output/Tensordot�
1autoencoder/decoder_output/BiasAdd/ReadVariableOpReadVariableOp:autoencoder_decoder_output_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype023
1autoencoder/decoder_output/BiasAdd/ReadVariableOp�
"autoencoder/decoder_output/BiasAddBiasAdd-autoencoder/decoder_output/Tensordot:output:09autoencoder/decoder_output/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������2$
"autoencoder/decoder_output/BiasAdd�
"autoencoder/decoder_output/SigmoidSigmoid+autoencoder/decoder_output/BiasAdd:output:0*
T0*,
_output_shapes
:����������2$
"autoencoder/decoder_output/Sigmoid�
IdentityIdentity&autoencoder/decoder_output/Sigmoid:y:09^autoencoder/decoder_hidden_layer1/BiasAdd/ReadVariableOp;^autoencoder/decoder_hidden_layer1/Tensordot/ReadVariableOp2^autoencoder/decoder_output/BiasAdd/ReadVariableOp4^autoencoder/decoder_output/Tensordot/ReadVariableOp9^autoencoder/encoder_hidden_layer1/BiasAdd/ReadVariableOp;^autoencoder/encoder_hidden_layer1/Tensordot/ReadVariableOp*^autoencoder/latent/BiasAdd/ReadVariableOp,^autoencoder/latent/Tensordot/ReadVariableOp*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:����������::::::::2t
8autoencoder/decoder_hidden_layer1/BiasAdd/ReadVariableOp8autoencoder/decoder_hidden_layer1/BiasAdd/ReadVariableOp2x
:autoencoder/decoder_hidden_layer1/Tensordot/ReadVariableOp:autoencoder/decoder_hidden_layer1/Tensordot/ReadVariableOp2f
1autoencoder/decoder_output/BiasAdd/ReadVariableOp1autoencoder/decoder_output/BiasAdd/ReadVariableOp2j
3autoencoder/decoder_output/Tensordot/ReadVariableOp3autoencoder/decoder_output/Tensordot/ReadVariableOp2t
8autoencoder/encoder_hidden_layer1/BiasAdd/ReadVariableOp8autoencoder/encoder_hidden_layer1/BiasAdd/ReadVariableOp2x
:autoencoder/encoder_hidden_layer1/Tensordot/ReadVariableOp:autoencoder/encoder_hidden_layer1/Tensordot/ReadVariableOp2V
)autoencoder/latent/BiasAdd/ReadVariableOp)autoencoder/latent/BiasAdd/ReadVariableOp2Z
+autoencoder/latent/Tensordot/ReadVariableOp+autoencoder/latent/Tensordot/ReadVariableOp:- )
'
_user_specified_nameencoder_input
�7
�
__inference__traced_save_20363
file_prefix;
7savev2_encoder_hidden_layer1_kernel_read_readvariableop9
5savev2_encoder_hidden_layer1_bias_read_readvariableop,
(savev2_latent_kernel_read_readvariableop*
&savev2_latent_bias_read_readvariableop;
7savev2_decoder_hidden_layer1_kernel_read_readvariableop9
5savev2_decoder_hidden_layer1_bias_read_readvariableop4
0savev2_decoder_output_kernel_read_readvariableop2
.savev2_decoder_output_bias_read_readvariableop+
'savev2_rmsprop_iter_read_readvariableop	,
(savev2_rmsprop_decay_read_readvariableop4
0savev2_rmsprop_learning_rate_read_readvariableop/
+savev2_rmsprop_momentum_read_readvariableop*
&savev2_rmsprop_rho_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableopG
Csavev2_rmsprop_encoder_hidden_layer1_kernel_rms_read_readvariableopE
Asavev2_rmsprop_encoder_hidden_layer1_bias_rms_read_readvariableop8
4savev2_rmsprop_latent_kernel_rms_read_readvariableop6
2savev2_rmsprop_latent_bias_rms_read_readvariableopG
Csavev2_rmsprop_decoder_hidden_layer1_kernel_rms_read_readvariableopE
Asavev2_rmsprop_decoder_hidden_layer1_bias_rms_read_readvariableop@
<savev2_rmsprop_decoder_output_kernel_rms_read_readvariableop>
:savev2_rmsprop_decoder_output_bias_rms_read_readvariableop
savev2_1_const

identity_1��MergeV2Checkpoints�SaveV2�SaveV2_1�
StringJoin/inputs_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_10241087a82f401d9c428536e2370b16/part2
StringJoin/inputs_1�

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:07savev2_encoder_hidden_layer1_kernel_read_readvariableop5savev2_encoder_hidden_layer1_bias_read_readvariableop(savev2_latent_kernel_read_readvariableop&savev2_latent_bias_read_readvariableop7savev2_decoder_hidden_layer1_kernel_read_readvariableop5savev2_decoder_hidden_layer1_bias_read_readvariableop0savev2_decoder_output_kernel_read_readvariableop.savev2_decoder_output_bias_read_readvariableop'savev2_rmsprop_iter_read_readvariableop(savev2_rmsprop_decay_read_readvariableop0savev2_rmsprop_learning_rate_read_readvariableop+savev2_rmsprop_momentum_read_readvariableop&savev2_rmsprop_rho_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopCsavev2_rmsprop_encoder_hidden_layer1_kernel_rms_read_readvariableopAsavev2_rmsprop_encoder_hidden_layer1_bias_rms_read_readvariableop4savev2_rmsprop_latent_kernel_rms_read_readvariableop2savev2_rmsprop_latent_bias_rms_read_readvariableopCsavev2_rmsprop_decoder_hidden_layer1_kernel_rms_read_readvariableopAsavev2_rmsprop_decoder_hidden_layer1_bias_rms_read_readvariableop<savev2_rmsprop_decoder_output_kernel_rms_read_readvariableop:savev2_rmsprop_decoder_output_bias_rms_read_readvariableop"/device:CPU:0*
_output_shapes
 *%
dtypes
2	2
SaveV2�
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard�
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1�
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names�
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices�
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity�

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*�
_input_shapes�
�: :	� : : :: : :	 �:�: : : : : : : :	� : : :: : :	 �:�: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix
�$
�
P__inference_encoder_hidden_layer1_layer_call_and_return_conditional_losses_19569

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp�
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	� *
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis�
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis�
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const�
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1�
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis�
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat�
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:����������2
Tensordot/transpose�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
Tensordot/Reshape�
Tensordot/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/transpose_1/perm�
Tensordot/transpose_1	Transpose Tensordot/ReadVariableOp:value:0#Tensordot/transpose_1/perm:output:0*
T0*
_output_shapes
:	� 2
Tensordot/transpose_1�
Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"�       2
Tensordot/Reshape_1/shape�
Tensordot/Reshape_1ReshapeTensordot/transpose_1:y:0"Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes
:	� 2
Tensordot/Reshape_1�
Tensordot/MatMulMatMulTensordot/Reshape:output:0Tensordot/Reshape_1:output:0*
T0*'
_output_shapes
:��������� 2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis�
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:��������� 2
	Tensordot�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:��������� 2	
BiasAdd\
TanhTanhBiasAdd:output:0*
T0*+
_output_shapes
:��������� 2
Tanh�
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:& "
 
_user_specified_nameinputs
�$
�
I__inference_decoder_output_layer_call_and_return_conditional_losses_19710

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp�
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	 �*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis�
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis�
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const�
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1�
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis�
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat�
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:��������� 2
Tensordot/transpose�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
Tensordot/Reshape�
Tensordot/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/transpose_1/perm�
Tensordot/transpose_1	Transpose Tensordot/ReadVariableOp:value:0#Tensordot/transpose_1/perm:output:0*
T0*
_output_shapes
:	 �2
Tensordot/transpose_1�
Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"    �   2
Tensordot/Reshape_1/shape�
Tensordot/Reshape_1ReshapeTensordot/transpose_1:y:0"Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes
:	 �2
Tensordot/Reshape_1�
Tensordot/MatMulMatMulTensordot/Reshape:output:0Tensordot/Reshape_1:output:0*
T0*(
_output_shapes
:����������2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis�
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:����������2
	Tensordot�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������2	
BiasAddf
SigmoidSigmoidBiasAdd:output:0*
T0*,
_output_shapes
:����������2	
Sigmoid�
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*2
_input_shapes!
:��������� ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:& "
 
_user_specified_nameinputs
�$
�
P__inference_decoder_hidden_layer1_layer_call_and_return_conditional_losses_19663

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp�
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

: *
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis�
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis�
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const�
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1�
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis�
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat�
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:���������2
Tensordot/transpose�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
Tensordot/Reshape�
Tensordot/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/transpose_1/perm�
Tensordot/transpose_1	Transpose Tensordot/ReadVariableOp:value:0#Tensordot/transpose_1/perm:output:0*
T0*
_output_shapes

: 2
Tensordot/transpose_1�
Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/Reshape_1/shape�
Tensordot/Reshape_1ReshapeTensordot/transpose_1:y:0"Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes

: 2
Tensordot/Reshape_1�
Tensordot/MatMulMatMulTensordot/Reshape:output:0Tensordot/Reshape_1:output:0*
T0*'
_output_shapes
:��������� 2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis�
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:��������� 2
	Tensordot�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:��������� 2	
BiasAdd\
TanhTanhBiasAdd:output:0*
T0*+
_output_shapes
:��������� 2
Tanh�
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:& "
 
_user_specified_nameinputs
��
�
F__inference_autoencoder_layer_call_and_return_conditional_losses_19948

inputs;
7encoder_hidden_layer1_tensordot_readvariableop_resource9
5encoder_hidden_layer1_biasadd_readvariableop_resource,
(latent_tensordot_readvariableop_resource*
&latent_biasadd_readvariableop_resource;
7decoder_hidden_layer1_tensordot_readvariableop_resource9
5decoder_hidden_layer1_biasadd_readvariableop_resource4
0decoder_output_tensordot_readvariableop_resource2
.decoder_output_biasadd_readvariableop_resource
identity��,decoder_hidden_layer1/BiasAdd/ReadVariableOp�.decoder_hidden_layer1/Tensordot/ReadVariableOp�%decoder_output/BiasAdd/ReadVariableOp�'decoder_output/Tensordot/ReadVariableOp�,encoder_hidden_layer1/BiasAdd/ReadVariableOp�.encoder_hidden_layer1/Tensordot/ReadVariableOp�latent/BiasAdd/ReadVariableOp�latent/Tensordot/ReadVariableOp�
.encoder_hidden_layer1/Tensordot/ReadVariableOpReadVariableOp7encoder_hidden_layer1_tensordot_readvariableop_resource*
_output_shapes
:	� *
dtype020
.encoder_hidden_layer1/Tensordot/ReadVariableOp�
$encoder_hidden_layer1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$encoder_hidden_layer1/Tensordot/axes�
$encoder_hidden_layer1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$encoder_hidden_layer1/Tensordot/free�
%encoder_hidden_layer1/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2'
%encoder_hidden_layer1/Tensordot/Shape�
-encoder_hidden_layer1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-encoder_hidden_layer1/Tensordot/GatherV2/axis�
(encoder_hidden_layer1/Tensordot/GatherV2GatherV2.encoder_hidden_layer1/Tensordot/Shape:output:0-encoder_hidden_layer1/Tensordot/free:output:06encoder_hidden_layer1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(encoder_hidden_layer1/Tensordot/GatherV2�
/encoder_hidden_layer1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/encoder_hidden_layer1/Tensordot/GatherV2_1/axis�
*encoder_hidden_layer1/Tensordot/GatherV2_1GatherV2.encoder_hidden_layer1/Tensordot/Shape:output:0-encoder_hidden_layer1/Tensordot/axes:output:08encoder_hidden_layer1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*encoder_hidden_layer1/Tensordot/GatherV2_1�
%encoder_hidden_layer1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%encoder_hidden_layer1/Tensordot/Const�
$encoder_hidden_layer1/Tensordot/ProdProd1encoder_hidden_layer1/Tensordot/GatherV2:output:0.encoder_hidden_layer1/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$encoder_hidden_layer1/Tensordot/Prod�
'encoder_hidden_layer1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'encoder_hidden_layer1/Tensordot/Const_1�
&encoder_hidden_layer1/Tensordot/Prod_1Prod3encoder_hidden_layer1/Tensordot/GatherV2_1:output:00encoder_hidden_layer1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&encoder_hidden_layer1/Tensordot/Prod_1�
+encoder_hidden_layer1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+encoder_hidden_layer1/Tensordot/concat/axis�
&encoder_hidden_layer1/Tensordot/concatConcatV2-encoder_hidden_layer1/Tensordot/free:output:0-encoder_hidden_layer1/Tensordot/axes:output:04encoder_hidden_layer1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&encoder_hidden_layer1/Tensordot/concat�
%encoder_hidden_layer1/Tensordot/stackPack-encoder_hidden_layer1/Tensordot/Prod:output:0/encoder_hidden_layer1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%encoder_hidden_layer1/Tensordot/stack�
)encoder_hidden_layer1/Tensordot/transpose	Transposeinputs/encoder_hidden_layer1/Tensordot/concat:output:0*
T0*,
_output_shapes
:����������2+
)encoder_hidden_layer1/Tensordot/transpose�
'encoder_hidden_layer1/Tensordot/ReshapeReshape-encoder_hidden_layer1/Tensordot/transpose:y:0.encoder_hidden_layer1/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2)
'encoder_hidden_layer1/Tensordot/Reshape�
0encoder_hidden_layer1/Tensordot/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       22
0encoder_hidden_layer1/Tensordot/transpose_1/perm�
+encoder_hidden_layer1/Tensordot/transpose_1	Transpose6encoder_hidden_layer1/Tensordot/ReadVariableOp:value:09encoder_hidden_layer1/Tensordot/transpose_1/perm:output:0*
T0*
_output_shapes
:	� 2-
+encoder_hidden_layer1/Tensordot/transpose_1�
/encoder_hidden_layer1/Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"�       21
/encoder_hidden_layer1/Tensordot/Reshape_1/shape�
)encoder_hidden_layer1/Tensordot/Reshape_1Reshape/encoder_hidden_layer1/Tensordot/transpose_1:y:08encoder_hidden_layer1/Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes
:	� 2+
)encoder_hidden_layer1/Tensordot/Reshape_1�
&encoder_hidden_layer1/Tensordot/MatMulMatMul0encoder_hidden_layer1/Tensordot/Reshape:output:02encoder_hidden_layer1/Tensordot/Reshape_1:output:0*
T0*'
_output_shapes
:��������� 2(
&encoder_hidden_layer1/Tensordot/MatMul�
'encoder_hidden_layer1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2)
'encoder_hidden_layer1/Tensordot/Const_2�
-encoder_hidden_layer1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-encoder_hidden_layer1/Tensordot/concat_1/axis�
(encoder_hidden_layer1/Tensordot/concat_1ConcatV21encoder_hidden_layer1/Tensordot/GatherV2:output:00encoder_hidden_layer1/Tensordot/Const_2:output:06encoder_hidden_layer1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(encoder_hidden_layer1/Tensordot/concat_1�
encoder_hidden_layer1/TensordotReshape0encoder_hidden_layer1/Tensordot/MatMul:product:01encoder_hidden_layer1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:��������� 2!
encoder_hidden_layer1/Tensordot�
,encoder_hidden_layer1/BiasAdd/ReadVariableOpReadVariableOp5encoder_hidden_layer1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,encoder_hidden_layer1/BiasAdd/ReadVariableOp�
encoder_hidden_layer1/BiasAddBiasAdd(encoder_hidden_layer1/Tensordot:output:04encoder_hidden_layer1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:��������� 2
encoder_hidden_layer1/BiasAdd�
encoder_hidden_layer1/TanhTanh&encoder_hidden_layer1/BiasAdd:output:0*
T0*+
_output_shapes
:��������� 2
encoder_hidden_layer1/Tanh�
latent/Tensordot/ReadVariableOpReadVariableOp(latent_tensordot_readvariableop_resource*
_output_shapes

: *
dtype02!
latent/Tensordot/ReadVariableOpx
latent/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
latent/Tensordot/axes
latent/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
latent/Tensordot/free~
latent/Tensordot/ShapeShapeencoder_hidden_layer1/Tanh:y:0*
T0*
_output_shapes
:2
latent/Tensordot/Shape�
latent/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
latent/Tensordot/GatherV2/axis�
latent/Tensordot/GatherV2GatherV2latent/Tensordot/Shape:output:0latent/Tensordot/free:output:0'latent/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
latent/Tensordot/GatherV2�
 latent/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 latent/Tensordot/GatherV2_1/axis�
latent/Tensordot/GatherV2_1GatherV2latent/Tensordot/Shape:output:0latent/Tensordot/axes:output:0)latent/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
latent/Tensordot/GatherV2_1z
latent/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
latent/Tensordot/Const�
latent/Tensordot/ProdProd"latent/Tensordot/GatherV2:output:0latent/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
latent/Tensordot/Prod~
latent/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
latent/Tensordot/Const_1�
latent/Tensordot/Prod_1Prod$latent/Tensordot/GatherV2_1:output:0!latent/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
latent/Tensordot/Prod_1~
latent/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
latent/Tensordot/concat/axis�
latent/Tensordot/concatConcatV2latent/Tensordot/free:output:0latent/Tensordot/axes:output:0%latent/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
latent/Tensordot/concat�
latent/Tensordot/stackPacklatent/Tensordot/Prod:output:0 latent/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
latent/Tensordot/stack�
latent/Tensordot/transpose	Transposeencoder_hidden_layer1/Tanh:y:0 latent/Tensordot/concat:output:0*
T0*+
_output_shapes
:��������� 2
latent/Tensordot/transpose�
latent/Tensordot/ReshapeReshapelatent/Tensordot/transpose:y:0latent/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
latent/Tensordot/Reshape�
!latent/Tensordot/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2#
!latent/Tensordot/transpose_1/perm�
latent/Tensordot/transpose_1	Transpose'latent/Tensordot/ReadVariableOp:value:0*latent/Tensordot/transpose_1/perm:output:0*
T0*
_output_shapes

: 2
latent/Tensordot/transpose_1�
 latent/Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"       2"
 latent/Tensordot/Reshape_1/shape�
latent/Tensordot/Reshape_1Reshape latent/Tensordot/transpose_1:y:0)latent/Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes

: 2
latent/Tensordot/Reshape_1�
latent/Tensordot/MatMulMatMul!latent/Tensordot/Reshape:output:0#latent/Tensordot/Reshape_1:output:0*
T0*'
_output_shapes
:���������2
latent/Tensordot/MatMul~
latent/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
latent/Tensordot/Const_2�
latent/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
latent/Tensordot/concat_1/axis�
latent/Tensordot/concat_1ConcatV2"latent/Tensordot/GatherV2:output:0!latent/Tensordot/Const_2:output:0'latent/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
latent/Tensordot/concat_1�
latent/TensordotReshape!latent/Tensordot/MatMul:product:0"latent/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������2
latent/Tensordot�
latent/BiasAdd/ReadVariableOpReadVariableOp&latent_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
latent/BiasAdd/ReadVariableOp�
latent/BiasAddBiasAddlatent/Tensordot:output:0%latent/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������2
latent/BiasAddq
latent/TanhTanhlatent/BiasAdd:output:0*
T0*+
_output_shapes
:���������2
latent/Tanh�
.decoder_hidden_layer1/Tensordot/ReadVariableOpReadVariableOp7decoder_hidden_layer1_tensordot_readvariableop_resource*
_output_shapes

: *
dtype020
.decoder_hidden_layer1/Tensordot/ReadVariableOp�
$decoder_hidden_layer1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$decoder_hidden_layer1/Tensordot/axes�
$decoder_hidden_layer1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$decoder_hidden_layer1/Tensordot/free�
%decoder_hidden_layer1/Tensordot/ShapeShapelatent/Tanh:y:0*
T0*
_output_shapes
:2'
%decoder_hidden_layer1/Tensordot/Shape�
-decoder_hidden_layer1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-decoder_hidden_layer1/Tensordot/GatherV2/axis�
(decoder_hidden_layer1/Tensordot/GatherV2GatherV2.decoder_hidden_layer1/Tensordot/Shape:output:0-decoder_hidden_layer1/Tensordot/free:output:06decoder_hidden_layer1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(decoder_hidden_layer1/Tensordot/GatherV2�
/decoder_hidden_layer1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/decoder_hidden_layer1/Tensordot/GatherV2_1/axis�
*decoder_hidden_layer1/Tensordot/GatherV2_1GatherV2.decoder_hidden_layer1/Tensordot/Shape:output:0-decoder_hidden_layer1/Tensordot/axes:output:08decoder_hidden_layer1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*decoder_hidden_layer1/Tensordot/GatherV2_1�
%decoder_hidden_layer1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%decoder_hidden_layer1/Tensordot/Const�
$decoder_hidden_layer1/Tensordot/ProdProd1decoder_hidden_layer1/Tensordot/GatherV2:output:0.decoder_hidden_layer1/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$decoder_hidden_layer1/Tensordot/Prod�
'decoder_hidden_layer1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'decoder_hidden_layer1/Tensordot/Const_1�
&decoder_hidden_layer1/Tensordot/Prod_1Prod3decoder_hidden_layer1/Tensordot/GatherV2_1:output:00decoder_hidden_layer1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&decoder_hidden_layer1/Tensordot/Prod_1�
+decoder_hidden_layer1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+decoder_hidden_layer1/Tensordot/concat/axis�
&decoder_hidden_layer1/Tensordot/concatConcatV2-decoder_hidden_layer1/Tensordot/free:output:0-decoder_hidden_layer1/Tensordot/axes:output:04decoder_hidden_layer1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&decoder_hidden_layer1/Tensordot/concat�
%decoder_hidden_layer1/Tensordot/stackPack-decoder_hidden_layer1/Tensordot/Prod:output:0/decoder_hidden_layer1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%decoder_hidden_layer1/Tensordot/stack�
)decoder_hidden_layer1/Tensordot/transpose	Transposelatent/Tanh:y:0/decoder_hidden_layer1/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������2+
)decoder_hidden_layer1/Tensordot/transpose�
'decoder_hidden_layer1/Tensordot/ReshapeReshape-decoder_hidden_layer1/Tensordot/transpose:y:0.decoder_hidden_layer1/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2)
'decoder_hidden_layer1/Tensordot/Reshape�
0decoder_hidden_layer1/Tensordot/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       22
0decoder_hidden_layer1/Tensordot/transpose_1/perm�
+decoder_hidden_layer1/Tensordot/transpose_1	Transpose6decoder_hidden_layer1/Tensordot/ReadVariableOp:value:09decoder_hidden_layer1/Tensordot/transpose_1/perm:output:0*
T0*
_output_shapes

: 2-
+decoder_hidden_layer1/Tensordot/transpose_1�
/decoder_hidden_layer1/Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"       21
/decoder_hidden_layer1/Tensordot/Reshape_1/shape�
)decoder_hidden_layer1/Tensordot/Reshape_1Reshape/decoder_hidden_layer1/Tensordot/transpose_1:y:08decoder_hidden_layer1/Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes

: 2+
)decoder_hidden_layer1/Tensordot/Reshape_1�
&decoder_hidden_layer1/Tensordot/MatMulMatMul0decoder_hidden_layer1/Tensordot/Reshape:output:02decoder_hidden_layer1/Tensordot/Reshape_1:output:0*
T0*'
_output_shapes
:��������� 2(
&decoder_hidden_layer1/Tensordot/MatMul�
'decoder_hidden_layer1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2)
'decoder_hidden_layer1/Tensordot/Const_2�
-decoder_hidden_layer1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-decoder_hidden_layer1/Tensordot/concat_1/axis�
(decoder_hidden_layer1/Tensordot/concat_1ConcatV21decoder_hidden_layer1/Tensordot/GatherV2:output:00decoder_hidden_layer1/Tensordot/Const_2:output:06decoder_hidden_layer1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(decoder_hidden_layer1/Tensordot/concat_1�
decoder_hidden_layer1/TensordotReshape0decoder_hidden_layer1/Tensordot/MatMul:product:01decoder_hidden_layer1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:��������� 2!
decoder_hidden_layer1/Tensordot�
,decoder_hidden_layer1/BiasAdd/ReadVariableOpReadVariableOp5decoder_hidden_layer1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,decoder_hidden_layer1/BiasAdd/ReadVariableOp�
decoder_hidden_layer1/BiasAddBiasAdd(decoder_hidden_layer1/Tensordot:output:04decoder_hidden_layer1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:��������� 2
decoder_hidden_layer1/BiasAdd�
decoder_hidden_layer1/TanhTanh&decoder_hidden_layer1/BiasAdd:output:0*
T0*+
_output_shapes
:��������� 2
decoder_hidden_layer1/Tanh�
'decoder_output/Tensordot/ReadVariableOpReadVariableOp0decoder_output_tensordot_readvariableop_resource*
_output_shapes
:	 �*
dtype02)
'decoder_output/Tensordot/ReadVariableOp�
decoder_output/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
decoder_output/Tensordot/axes�
decoder_output/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
decoder_output/Tensordot/free�
decoder_output/Tensordot/ShapeShapedecoder_hidden_layer1/Tanh:y:0*
T0*
_output_shapes
:2 
decoder_output/Tensordot/Shape�
&decoder_output/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2(
&decoder_output/Tensordot/GatherV2/axis�
!decoder_output/Tensordot/GatherV2GatherV2'decoder_output/Tensordot/Shape:output:0&decoder_output/Tensordot/free:output:0/decoder_output/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2#
!decoder_output/Tensordot/GatherV2�
(decoder_output/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2*
(decoder_output/Tensordot/GatherV2_1/axis�
#decoder_output/Tensordot/GatherV2_1GatherV2'decoder_output/Tensordot/Shape:output:0&decoder_output/Tensordot/axes:output:01decoder_output/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2%
#decoder_output/Tensordot/GatherV2_1�
decoder_output/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
decoder_output/Tensordot/Const�
decoder_output/Tensordot/ProdProd*decoder_output/Tensordot/GatherV2:output:0'decoder_output/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
decoder_output/Tensordot/Prod�
 decoder_output/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 decoder_output/Tensordot/Const_1�
decoder_output/Tensordot/Prod_1Prod,decoder_output/Tensordot/GatherV2_1:output:0)decoder_output/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2!
decoder_output/Tensordot/Prod_1�
$decoder_output/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$decoder_output/Tensordot/concat/axis�
decoder_output/Tensordot/concatConcatV2&decoder_output/Tensordot/free:output:0&decoder_output/Tensordot/axes:output:0-decoder_output/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2!
decoder_output/Tensordot/concat�
decoder_output/Tensordot/stackPack&decoder_output/Tensordot/Prod:output:0(decoder_output/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2 
decoder_output/Tensordot/stack�
"decoder_output/Tensordot/transpose	Transposedecoder_hidden_layer1/Tanh:y:0(decoder_output/Tensordot/concat:output:0*
T0*+
_output_shapes
:��������� 2$
"decoder_output/Tensordot/transpose�
 decoder_output/Tensordot/ReshapeReshape&decoder_output/Tensordot/transpose:y:0'decoder_output/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2"
 decoder_output/Tensordot/Reshape�
)decoder_output/Tensordot/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2+
)decoder_output/Tensordot/transpose_1/perm�
$decoder_output/Tensordot/transpose_1	Transpose/decoder_output/Tensordot/ReadVariableOp:value:02decoder_output/Tensordot/transpose_1/perm:output:0*
T0*
_output_shapes
:	 �2&
$decoder_output/Tensordot/transpose_1�
(decoder_output/Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"    �   2*
(decoder_output/Tensordot/Reshape_1/shape�
"decoder_output/Tensordot/Reshape_1Reshape(decoder_output/Tensordot/transpose_1:y:01decoder_output/Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes
:	 �2$
"decoder_output/Tensordot/Reshape_1�
decoder_output/Tensordot/MatMulMatMul)decoder_output/Tensordot/Reshape:output:0+decoder_output/Tensordot/Reshape_1:output:0*
T0*(
_output_shapes
:����������2!
decoder_output/Tensordot/MatMul�
 decoder_output/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�2"
 decoder_output/Tensordot/Const_2�
&decoder_output/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2(
&decoder_output/Tensordot/concat_1/axis�
!decoder_output/Tensordot/concat_1ConcatV2*decoder_output/Tensordot/GatherV2:output:0)decoder_output/Tensordot/Const_2:output:0/decoder_output/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2#
!decoder_output/Tensordot/concat_1�
decoder_output/TensordotReshape)decoder_output/Tensordot/MatMul:product:0*decoder_output/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:����������2
decoder_output/Tensordot�
%decoder_output/BiasAdd/ReadVariableOpReadVariableOp.decoder_output_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02'
%decoder_output/BiasAdd/ReadVariableOp�
decoder_output/BiasAddBiasAdd!decoder_output/Tensordot:output:0-decoder_output/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������2
decoder_output/BiasAdd�
decoder_output/SigmoidSigmoiddecoder_output/BiasAdd:output:0*
T0*,
_output_shapes
:����������2
decoder_output/Sigmoid�
IdentityIdentitydecoder_output/Sigmoid:y:0-^decoder_hidden_layer1/BiasAdd/ReadVariableOp/^decoder_hidden_layer1/Tensordot/ReadVariableOp&^decoder_output/BiasAdd/ReadVariableOp(^decoder_output/Tensordot/ReadVariableOp-^encoder_hidden_layer1/BiasAdd/ReadVariableOp/^encoder_hidden_layer1/Tensordot/ReadVariableOp^latent/BiasAdd/ReadVariableOp ^latent/Tensordot/ReadVariableOp*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:����������::::::::2\
,decoder_hidden_layer1/BiasAdd/ReadVariableOp,decoder_hidden_layer1/BiasAdd/ReadVariableOp2`
.decoder_hidden_layer1/Tensordot/ReadVariableOp.decoder_hidden_layer1/Tensordot/ReadVariableOp2N
%decoder_output/BiasAdd/ReadVariableOp%decoder_output/BiasAdd/ReadVariableOp2R
'decoder_output/Tensordot/ReadVariableOp'decoder_output/Tensordot/ReadVariableOp2\
,encoder_hidden_layer1/BiasAdd/ReadVariableOp,encoder_hidden_layer1/BiasAdd/ReadVariableOp2`
.encoder_hidden_layer1/Tensordot/ReadVariableOp.encoder_hidden_layer1/Tensordot/ReadVariableOp2>
latent/BiasAdd/ReadVariableOplatent/BiasAdd/ReadVariableOp2B
latent/Tensordot/ReadVariableOplatent/Tensordot/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
F__inference_autoencoder_layer_call_and_return_conditional_losses_19787

inputs8
4encoder_hidden_layer1_statefulpartitionedcall_args_18
4encoder_hidden_layer1_statefulpartitionedcall_args_2)
%latent_statefulpartitionedcall_args_1)
%latent_statefulpartitionedcall_args_28
4decoder_hidden_layer1_statefulpartitionedcall_args_18
4decoder_hidden_layer1_statefulpartitionedcall_args_21
-decoder_output_statefulpartitionedcall_args_11
-decoder_output_statefulpartitionedcall_args_2
identity��-decoder_hidden_layer1/StatefulPartitionedCall�&decoder_output/StatefulPartitionedCall�-encoder_hidden_layer1/StatefulPartitionedCall�latent/StatefulPartitionedCall�
-encoder_hidden_layer1/StatefulPartitionedCallStatefulPartitionedCallinputs4encoder_hidden_layer1_statefulpartitionedcall_args_14encoder_hidden_layer1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*+
_output_shapes
:��������� *-
config_proto

CPU

GPU2*0J 8*Y
fTRR
P__inference_encoder_hidden_layer1_layer_call_and_return_conditional_losses_195692/
-encoder_hidden_layer1/StatefulPartitionedCall�
latent/StatefulPartitionedCallStatefulPartitionedCall6encoder_hidden_layer1/StatefulPartitionedCall:output:0%latent_statefulpartitionedcall_args_1%latent_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*+
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_latent_layer_call_and_return_conditional_losses_196162 
latent/StatefulPartitionedCall�
-decoder_hidden_layer1/StatefulPartitionedCallStatefulPartitionedCall'latent/StatefulPartitionedCall:output:04decoder_hidden_layer1_statefulpartitionedcall_args_14decoder_hidden_layer1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*+
_output_shapes
:��������� *-
config_proto

CPU

GPU2*0J 8*Y
fTRR
P__inference_decoder_hidden_layer1_layer_call_and_return_conditional_losses_196632/
-decoder_hidden_layer1/StatefulPartitionedCall�
&decoder_output/StatefulPartitionedCallStatefulPartitionedCall6decoder_hidden_layer1/StatefulPartitionedCall:output:0-decoder_output_statefulpartitionedcall_args_1-decoder_output_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*,
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_decoder_output_layer_call_and_return_conditional_losses_197102(
&decoder_output/StatefulPartitionedCall�
IdentityIdentity/decoder_output/StatefulPartitionedCall:output:0.^decoder_hidden_layer1/StatefulPartitionedCall'^decoder_output/StatefulPartitionedCall.^encoder_hidden_layer1/StatefulPartitionedCall^latent/StatefulPartitionedCall*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:����������::::::::2^
-decoder_hidden_layer1/StatefulPartitionedCall-decoder_hidden_layer1/StatefulPartitionedCall2P
&decoder_output/StatefulPartitionedCall&decoder_output/StatefulPartitionedCall2^
-encoder_hidden_layer1/StatefulPartitionedCall-encoder_hidden_layer1/StatefulPartitionedCall2@
latent/StatefulPartitionedCalllatent/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
F__inference_autoencoder_layer_call_and_return_conditional_losses_19723
encoder_input8
4encoder_hidden_layer1_statefulpartitionedcall_args_18
4encoder_hidden_layer1_statefulpartitionedcall_args_2)
%latent_statefulpartitionedcall_args_1)
%latent_statefulpartitionedcall_args_28
4decoder_hidden_layer1_statefulpartitionedcall_args_18
4decoder_hidden_layer1_statefulpartitionedcall_args_21
-decoder_output_statefulpartitionedcall_args_11
-decoder_output_statefulpartitionedcall_args_2
identity��-decoder_hidden_layer1/StatefulPartitionedCall�&decoder_output/StatefulPartitionedCall�-encoder_hidden_layer1/StatefulPartitionedCall�latent/StatefulPartitionedCall�
-encoder_hidden_layer1/StatefulPartitionedCallStatefulPartitionedCallencoder_input4encoder_hidden_layer1_statefulpartitionedcall_args_14encoder_hidden_layer1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*+
_output_shapes
:��������� *-
config_proto

CPU

GPU2*0J 8*Y
fTRR
P__inference_encoder_hidden_layer1_layer_call_and_return_conditional_losses_195692/
-encoder_hidden_layer1/StatefulPartitionedCall�
latent/StatefulPartitionedCallStatefulPartitionedCall6encoder_hidden_layer1/StatefulPartitionedCall:output:0%latent_statefulpartitionedcall_args_1%latent_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*+
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_latent_layer_call_and_return_conditional_losses_196162 
latent/StatefulPartitionedCall�
-decoder_hidden_layer1/StatefulPartitionedCallStatefulPartitionedCall'latent/StatefulPartitionedCall:output:04decoder_hidden_layer1_statefulpartitionedcall_args_14decoder_hidden_layer1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*+
_output_shapes
:��������� *-
config_proto

CPU

GPU2*0J 8*Y
fTRR
P__inference_decoder_hidden_layer1_layer_call_and_return_conditional_losses_196632/
-decoder_hidden_layer1/StatefulPartitionedCall�
&decoder_output/StatefulPartitionedCallStatefulPartitionedCall6decoder_hidden_layer1/StatefulPartitionedCall:output:0-decoder_output_statefulpartitionedcall_args_1-decoder_output_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*,
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_decoder_output_layer_call_and_return_conditional_losses_197102(
&decoder_output/StatefulPartitionedCall�
IdentityIdentity/decoder_output/StatefulPartitionedCall:output:0.^decoder_hidden_layer1/StatefulPartitionedCall'^decoder_output/StatefulPartitionedCall.^encoder_hidden_layer1/StatefulPartitionedCall^latent/StatefulPartitionedCall*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:����������::::::::2^
-decoder_hidden_layer1/StatefulPartitionedCall-decoder_hidden_layer1/StatefulPartitionedCall2P
&decoder_output/StatefulPartitionedCall&decoder_output/StatefulPartitionedCall2^
-encoder_hidden_layer1/StatefulPartitionedCall-encoder_hidden_layer1/StatefulPartitionedCall2@
latent/StatefulPartitionedCalllatent/StatefulPartitionedCall:- )
'
_user_specified_nameencoder_input
�$
�
P__inference_decoder_hidden_layer1_layer_call_and_return_conditional_losses_20221

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp�
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

: *
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis�
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis�
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const�
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1�
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis�
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat�
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:���������2
Tensordot/transpose�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
Tensordot/Reshape�
Tensordot/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/transpose_1/perm�
Tensordot/transpose_1	Transpose Tensordot/ReadVariableOp:value:0#Tensordot/transpose_1/perm:output:0*
T0*
_output_shapes

: 2
Tensordot/transpose_1�
Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/Reshape_1/shape�
Tensordot/Reshape_1ReshapeTensordot/transpose_1:y:0"Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes

: 2
Tensordot/Reshape_1�
Tensordot/MatMulMatMulTensordot/Reshape:output:0Tensordot/Reshape_1:output:0*
T0*'
_output_shapes
:��������� 2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis�
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:��������� 2
	Tensordot�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:��������� 2	
BiasAdd\
TanhTanhBiasAdd:output:0*
T0*+
_output_shapes
:��������� 2
Tanh�
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:& "
 
_user_specified_nameinputs
�

�
+__inference_autoencoder_layer_call_fn_19769
encoder_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallencoder_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*
Tin
2	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*,
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_autoencoder_layer_call_and_return_conditional_losses_197582
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:����������::::::::22
StatefulPartitionedCallStatefulPartitionedCall:- )
'
_user_specified_nameencoder_input
�

�
+__inference_autoencoder_layer_call_fn_20089

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*
Tin
2	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*,
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_autoencoder_layer_call_and_return_conditional_losses_197582
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:����������::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
F__inference_autoencoder_layer_call_and_return_conditional_losses_19758

inputs8
4encoder_hidden_layer1_statefulpartitionedcall_args_18
4encoder_hidden_layer1_statefulpartitionedcall_args_2)
%latent_statefulpartitionedcall_args_1)
%latent_statefulpartitionedcall_args_28
4decoder_hidden_layer1_statefulpartitionedcall_args_18
4decoder_hidden_layer1_statefulpartitionedcall_args_21
-decoder_output_statefulpartitionedcall_args_11
-decoder_output_statefulpartitionedcall_args_2
identity��-decoder_hidden_layer1/StatefulPartitionedCall�&decoder_output/StatefulPartitionedCall�-encoder_hidden_layer1/StatefulPartitionedCall�latent/StatefulPartitionedCall�
-encoder_hidden_layer1/StatefulPartitionedCallStatefulPartitionedCallinputs4encoder_hidden_layer1_statefulpartitionedcall_args_14encoder_hidden_layer1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*+
_output_shapes
:��������� *-
config_proto

CPU

GPU2*0J 8*Y
fTRR
P__inference_encoder_hidden_layer1_layer_call_and_return_conditional_losses_195692/
-encoder_hidden_layer1/StatefulPartitionedCall�
latent/StatefulPartitionedCallStatefulPartitionedCall6encoder_hidden_layer1/StatefulPartitionedCall:output:0%latent_statefulpartitionedcall_args_1%latent_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*+
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_latent_layer_call_and_return_conditional_losses_196162 
latent/StatefulPartitionedCall�
-decoder_hidden_layer1/StatefulPartitionedCallStatefulPartitionedCall'latent/StatefulPartitionedCall:output:04decoder_hidden_layer1_statefulpartitionedcall_args_14decoder_hidden_layer1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*+
_output_shapes
:��������� *-
config_proto

CPU

GPU2*0J 8*Y
fTRR
P__inference_decoder_hidden_layer1_layer_call_and_return_conditional_losses_196632/
-decoder_hidden_layer1/StatefulPartitionedCall�
&decoder_output/StatefulPartitionedCallStatefulPartitionedCall6decoder_hidden_layer1/StatefulPartitionedCall:output:0-decoder_output_statefulpartitionedcall_args_1-decoder_output_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*,
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_decoder_output_layer_call_and_return_conditional_losses_197102(
&decoder_output/StatefulPartitionedCall�
IdentityIdentity/decoder_output/StatefulPartitionedCall:output:0.^decoder_hidden_layer1/StatefulPartitionedCall'^decoder_output/StatefulPartitionedCall.^encoder_hidden_layer1/StatefulPartitionedCall^latent/StatefulPartitionedCall*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:����������::::::::2^
-decoder_hidden_layer1/StatefulPartitionedCall-decoder_hidden_layer1/StatefulPartitionedCall2P
&decoder_output/StatefulPartitionedCall&decoder_output/StatefulPartitionedCall2^
-encoder_hidden_layer1/StatefulPartitionedCall-encoder_hidden_layer1/StatefulPartitionedCall2@
latent/StatefulPartitionedCalllatent/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
�

�
#__inference_signature_wrapper_19820
encoder_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallencoder_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*
Tin
2	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*,
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*)
f$R"
 __inference__wrapped_model_195302
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:����������::::::::22
StatefulPartitionedCallStatefulPartitionedCall:- )
'
_user_specified_nameencoder_input
�$
�
A__inference_latent_layer_call_and_return_conditional_losses_19616

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp�
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

: *
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis�
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis�
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const�
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1�
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis�
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat�
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:��������� 2
Tensordot/transpose�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
Tensordot/Reshape�
Tensordot/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/transpose_1/perm�
Tensordot/transpose_1	Transpose Tensordot/ReadVariableOp:value:0#Tensordot/transpose_1/perm:output:0*
T0*
_output_shapes

: 2
Tensordot/transpose_1�
Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/Reshape_1/shape�
Tensordot/Reshape_1ReshapeTensordot/transpose_1:y:0"Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes

: 2
Tensordot/Reshape_1�
Tensordot/MatMulMatMulTensordot/Reshape:output:0Tensordot/Reshape_1:output:0*
T0*'
_output_shapes
:���������2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis�
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������2
	Tensordot�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������2	
BiasAdd\
TanhTanhBiasAdd:output:0*
T0*+
_output_shapes
:���������2
Tanh�
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:���������2

Identity"
identityIdentity:output:0*2
_input_shapes!
:��������� ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:& "
 
_user_specified_nameinputs
��
�
F__inference_autoencoder_layer_call_and_return_conditional_losses_20076

inputs;
7encoder_hidden_layer1_tensordot_readvariableop_resource9
5encoder_hidden_layer1_biasadd_readvariableop_resource,
(latent_tensordot_readvariableop_resource*
&latent_biasadd_readvariableop_resource;
7decoder_hidden_layer1_tensordot_readvariableop_resource9
5decoder_hidden_layer1_biasadd_readvariableop_resource4
0decoder_output_tensordot_readvariableop_resource2
.decoder_output_biasadd_readvariableop_resource
identity��,decoder_hidden_layer1/BiasAdd/ReadVariableOp�.decoder_hidden_layer1/Tensordot/ReadVariableOp�%decoder_output/BiasAdd/ReadVariableOp�'decoder_output/Tensordot/ReadVariableOp�,encoder_hidden_layer1/BiasAdd/ReadVariableOp�.encoder_hidden_layer1/Tensordot/ReadVariableOp�latent/BiasAdd/ReadVariableOp�latent/Tensordot/ReadVariableOp�
.encoder_hidden_layer1/Tensordot/ReadVariableOpReadVariableOp7encoder_hidden_layer1_tensordot_readvariableop_resource*
_output_shapes
:	� *
dtype020
.encoder_hidden_layer1/Tensordot/ReadVariableOp�
$encoder_hidden_layer1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$encoder_hidden_layer1/Tensordot/axes�
$encoder_hidden_layer1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$encoder_hidden_layer1/Tensordot/free�
%encoder_hidden_layer1/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2'
%encoder_hidden_layer1/Tensordot/Shape�
-encoder_hidden_layer1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-encoder_hidden_layer1/Tensordot/GatherV2/axis�
(encoder_hidden_layer1/Tensordot/GatherV2GatherV2.encoder_hidden_layer1/Tensordot/Shape:output:0-encoder_hidden_layer1/Tensordot/free:output:06encoder_hidden_layer1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(encoder_hidden_layer1/Tensordot/GatherV2�
/encoder_hidden_layer1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/encoder_hidden_layer1/Tensordot/GatherV2_1/axis�
*encoder_hidden_layer1/Tensordot/GatherV2_1GatherV2.encoder_hidden_layer1/Tensordot/Shape:output:0-encoder_hidden_layer1/Tensordot/axes:output:08encoder_hidden_layer1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*encoder_hidden_layer1/Tensordot/GatherV2_1�
%encoder_hidden_layer1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%encoder_hidden_layer1/Tensordot/Const�
$encoder_hidden_layer1/Tensordot/ProdProd1encoder_hidden_layer1/Tensordot/GatherV2:output:0.encoder_hidden_layer1/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$encoder_hidden_layer1/Tensordot/Prod�
'encoder_hidden_layer1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'encoder_hidden_layer1/Tensordot/Const_1�
&encoder_hidden_layer1/Tensordot/Prod_1Prod3encoder_hidden_layer1/Tensordot/GatherV2_1:output:00encoder_hidden_layer1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&encoder_hidden_layer1/Tensordot/Prod_1�
+encoder_hidden_layer1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+encoder_hidden_layer1/Tensordot/concat/axis�
&encoder_hidden_layer1/Tensordot/concatConcatV2-encoder_hidden_layer1/Tensordot/free:output:0-encoder_hidden_layer1/Tensordot/axes:output:04encoder_hidden_layer1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&encoder_hidden_layer1/Tensordot/concat�
%encoder_hidden_layer1/Tensordot/stackPack-encoder_hidden_layer1/Tensordot/Prod:output:0/encoder_hidden_layer1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%encoder_hidden_layer1/Tensordot/stack�
)encoder_hidden_layer1/Tensordot/transpose	Transposeinputs/encoder_hidden_layer1/Tensordot/concat:output:0*
T0*,
_output_shapes
:����������2+
)encoder_hidden_layer1/Tensordot/transpose�
'encoder_hidden_layer1/Tensordot/ReshapeReshape-encoder_hidden_layer1/Tensordot/transpose:y:0.encoder_hidden_layer1/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2)
'encoder_hidden_layer1/Tensordot/Reshape�
0encoder_hidden_layer1/Tensordot/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       22
0encoder_hidden_layer1/Tensordot/transpose_1/perm�
+encoder_hidden_layer1/Tensordot/transpose_1	Transpose6encoder_hidden_layer1/Tensordot/ReadVariableOp:value:09encoder_hidden_layer1/Tensordot/transpose_1/perm:output:0*
T0*
_output_shapes
:	� 2-
+encoder_hidden_layer1/Tensordot/transpose_1�
/encoder_hidden_layer1/Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"�       21
/encoder_hidden_layer1/Tensordot/Reshape_1/shape�
)encoder_hidden_layer1/Tensordot/Reshape_1Reshape/encoder_hidden_layer1/Tensordot/transpose_1:y:08encoder_hidden_layer1/Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes
:	� 2+
)encoder_hidden_layer1/Tensordot/Reshape_1�
&encoder_hidden_layer1/Tensordot/MatMulMatMul0encoder_hidden_layer1/Tensordot/Reshape:output:02encoder_hidden_layer1/Tensordot/Reshape_1:output:0*
T0*'
_output_shapes
:��������� 2(
&encoder_hidden_layer1/Tensordot/MatMul�
'encoder_hidden_layer1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2)
'encoder_hidden_layer1/Tensordot/Const_2�
-encoder_hidden_layer1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-encoder_hidden_layer1/Tensordot/concat_1/axis�
(encoder_hidden_layer1/Tensordot/concat_1ConcatV21encoder_hidden_layer1/Tensordot/GatherV2:output:00encoder_hidden_layer1/Tensordot/Const_2:output:06encoder_hidden_layer1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(encoder_hidden_layer1/Tensordot/concat_1�
encoder_hidden_layer1/TensordotReshape0encoder_hidden_layer1/Tensordot/MatMul:product:01encoder_hidden_layer1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:��������� 2!
encoder_hidden_layer1/Tensordot�
,encoder_hidden_layer1/BiasAdd/ReadVariableOpReadVariableOp5encoder_hidden_layer1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,encoder_hidden_layer1/BiasAdd/ReadVariableOp�
encoder_hidden_layer1/BiasAddBiasAdd(encoder_hidden_layer1/Tensordot:output:04encoder_hidden_layer1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:��������� 2
encoder_hidden_layer1/BiasAdd�
encoder_hidden_layer1/TanhTanh&encoder_hidden_layer1/BiasAdd:output:0*
T0*+
_output_shapes
:��������� 2
encoder_hidden_layer1/Tanh�
latent/Tensordot/ReadVariableOpReadVariableOp(latent_tensordot_readvariableop_resource*
_output_shapes

: *
dtype02!
latent/Tensordot/ReadVariableOpx
latent/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
latent/Tensordot/axes
latent/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
latent/Tensordot/free~
latent/Tensordot/ShapeShapeencoder_hidden_layer1/Tanh:y:0*
T0*
_output_shapes
:2
latent/Tensordot/Shape�
latent/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
latent/Tensordot/GatherV2/axis�
latent/Tensordot/GatherV2GatherV2latent/Tensordot/Shape:output:0latent/Tensordot/free:output:0'latent/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
latent/Tensordot/GatherV2�
 latent/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 latent/Tensordot/GatherV2_1/axis�
latent/Tensordot/GatherV2_1GatherV2latent/Tensordot/Shape:output:0latent/Tensordot/axes:output:0)latent/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
latent/Tensordot/GatherV2_1z
latent/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
latent/Tensordot/Const�
latent/Tensordot/ProdProd"latent/Tensordot/GatherV2:output:0latent/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
latent/Tensordot/Prod~
latent/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
latent/Tensordot/Const_1�
latent/Tensordot/Prod_1Prod$latent/Tensordot/GatherV2_1:output:0!latent/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
latent/Tensordot/Prod_1~
latent/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
latent/Tensordot/concat/axis�
latent/Tensordot/concatConcatV2latent/Tensordot/free:output:0latent/Tensordot/axes:output:0%latent/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
latent/Tensordot/concat�
latent/Tensordot/stackPacklatent/Tensordot/Prod:output:0 latent/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
latent/Tensordot/stack�
latent/Tensordot/transpose	Transposeencoder_hidden_layer1/Tanh:y:0 latent/Tensordot/concat:output:0*
T0*+
_output_shapes
:��������� 2
latent/Tensordot/transpose�
latent/Tensordot/ReshapeReshapelatent/Tensordot/transpose:y:0latent/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
latent/Tensordot/Reshape�
!latent/Tensordot/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2#
!latent/Tensordot/transpose_1/perm�
latent/Tensordot/transpose_1	Transpose'latent/Tensordot/ReadVariableOp:value:0*latent/Tensordot/transpose_1/perm:output:0*
T0*
_output_shapes

: 2
latent/Tensordot/transpose_1�
 latent/Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"       2"
 latent/Tensordot/Reshape_1/shape�
latent/Tensordot/Reshape_1Reshape latent/Tensordot/transpose_1:y:0)latent/Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes

: 2
latent/Tensordot/Reshape_1�
latent/Tensordot/MatMulMatMul!latent/Tensordot/Reshape:output:0#latent/Tensordot/Reshape_1:output:0*
T0*'
_output_shapes
:���������2
latent/Tensordot/MatMul~
latent/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
latent/Tensordot/Const_2�
latent/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
latent/Tensordot/concat_1/axis�
latent/Tensordot/concat_1ConcatV2"latent/Tensordot/GatherV2:output:0!latent/Tensordot/Const_2:output:0'latent/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
latent/Tensordot/concat_1�
latent/TensordotReshape!latent/Tensordot/MatMul:product:0"latent/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������2
latent/Tensordot�
latent/BiasAdd/ReadVariableOpReadVariableOp&latent_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
latent/BiasAdd/ReadVariableOp�
latent/BiasAddBiasAddlatent/Tensordot:output:0%latent/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������2
latent/BiasAddq
latent/TanhTanhlatent/BiasAdd:output:0*
T0*+
_output_shapes
:���������2
latent/Tanh�
.decoder_hidden_layer1/Tensordot/ReadVariableOpReadVariableOp7decoder_hidden_layer1_tensordot_readvariableop_resource*
_output_shapes

: *
dtype020
.decoder_hidden_layer1/Tensordot/ReadVariableOp�
$decoder_hidden_layer1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$decoder_hidden_layer1/Tensordot/axes�
$decoder_hidden_layer1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$decoder_hidden_layer1/Tensordot/free�
%decoder_hidden_layer1/Tensordot/ShapeShapelatent/Tanh:y:0*
T0*
_output_shapes
:2'
%decoder_hidden_layer1/Tensordot/Shape�
-decoder_hidden_layer1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-decoder_hidden_layer1/Tensordot/GatherV2/axis�
(decoder_hidden_layer1/Tensordot/GatherV2GatherV2.decoder_hidden_layer1/Tensordot/Shape:output:0-decoder_hidden_layer1/Tensordot/free:output:06decoder_hidden_layer1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(decoder_hidden_layer1/Tensordot/GatherV2�
/decoder_hidden_layer1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/decoder_hidden_layer1/Tensordot/GatherV2_1/axis�
*decoder_hidden_layer1/Tensordot/GatherV2_1GatherV2.decoder_hidden_layer1/Tensordot/Shape:output:0-decoder_hidden_layer1/Tensordot/axes:output:08decoder_hidden_layer1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*decoder_hidden_layer1/Tensordot/GatherV2_1�
%decoder_hidden_layer1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%decoder_hidden_layer1/Tensordot/Const�
$decoder_hidden_layer1/Tensordot/ProdProd1decoder_hidden_layer1/Tensordot/GatherV2:output:0.decoder_hidden_layer1/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$decoder_hidden_layer1/Tensordot/Prod�
'decoder_hidden_layer1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'decoder_hidden_layer1/Tensordot/Const_1�
&decoder_hidden_layer1/Tensordot/Prod_1Prod3decoder_hidden_layer1/Tensordot/GatherV2_1:output:00decoder_hidden_layer1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&decoder_hidden_layer1/Tensordot/Prod_1�
+decoder_hidden_layer1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+decoder_hidden_layer1/Tensordot/concat/axis�
&decoder_hidden_layer1/Tensordot/concatConcatV2-decoder_hidden_layer1/Tensordot/free:output:0-decoder_hidden_layer1/Tensordot/axes:output:04decoder_hidden_layer1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&decoder_hidden_layer1/Tensordot/concat�
%decoder_hidden_layer1/Tensordot/stackPack-decoder_hidden_layer1/Tensordot/Prod:output:0/decoder_hidden_layer1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%decoder_hidden_layer1/Tensordot/stack�
)decoder_hidden_layer1/Tensordot/transpose	Transposelatent/Tanh:y:0/decoder_hidden_layer1/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������2+
)decoder_hidden_layer1/Tensordot/transpose�
'decoder_hidden_layer1/Tensordot/ReshapeReshape-decoder_hidden_layer1/Tensordot/transpose:y:0.decoder_hidden_layer1/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2)
'decoder_hidden_layer1/Tensordot/Reshape�
0decoder_hidden_layer1/Tensordot/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       22
0decoder_hidden_layer1/Tensordot/transpose_1/perm�
+decoder_hidden_layer1/Tensordot/transpose_1	Transpose6decoder_hidden_layer1/Tensordot/ReadVariableOp:value:09decoder_hidden_layer1/Tensordot/transpose_1/perm:output:0*
T0*
_output_shapes

: 2-
+decoder_hidden_layer1/Tensordot/transpose_1�
/decoder_hidden_layer1/Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"       21
/decoder_hidden_layer1/Tensordot/Reshape_1/shape�
)decoder_hidden_layer1/Tensordot/Reshape_1Reshape/decoder_hidden_layer1/Tensordot/transpose_1:y:08decoder_hidden_layer1/Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes

: 2+
)decoder_hidden_layer1/Tensordot/Reshape_1�
&decoder_hidden_layer1/Tensordot/MatMulMatMul0decoder_hidden_layer1/Tensordot/Reshape:output:02decoder_hidden_layer1/Tensordot/Reshape_1:output:0*
T0*'
_output_shapes
:��������� 2(
&decoder_hidden_layer1/Tensordot/MatMul�
'decoder_hidden_layer1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2)
'decoder_hidden_layer1/Tensordot/Const_2�
-decoder_hidden_layer1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-decoder_hidden_layer1/Tensordot/concat_1/axis�
(decoder_hidden_layer1/Tensordot/concat_1ConcatV21decoder_hidden_layer1/Tensordot/GatherV2:output:00decoder_hidden_layer1/Tensordot/Const_2:output:06decoder_hidden_layer1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(decoder_hidden_layer1/Tensordot/concat_1�
decoder_hidden_layer1/TensordotReshape0decoder_hidden_layer1/Tensordot/MatMul:product:01decoder_hidden_layer1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:��������� 2!
decoder_hidden_layer1/Tensordot�
,decoder_hidden_layer1/BiasAdd/ReadVariableOpReadVariableOp5decoder_hidden_layer1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,decoder_hidden_layer1/BiasAdd/ReadVariableOp�
decoder_hidden_layer1/BiasAddBiasAdd(decoder_hidden_layer1/Tensordot:output:04decoder_hidden_layer1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:��������� 2
decoder_hidden_layer1/BiasAdd�
decoder_hidden_layer1/TanhTanh&decoder_hidden_layer1/BiasAdd:output:0*
T0*+
_output_shapes
:��������� 2
decoder_hidden_layer1/Tanh�
'decoder_output/Tensordot/ReadVariableOpReadVariableOp0decoder_output_tensordot_readvariableop_resource*
_output_shapes
:	 �*
dtype02)
'decoder_output/Tensordot/ReadVariableOp�
decoder_output/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
decoder_output/Tensordot/axes�
decoder_output/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
decoder_output/Tensordot/free�
decoder_output/Tensordot/ShapeShapedecoder_hidden_layer1/Tanh:y:0*
T0*
_output_shapes
:2 
decoder_output/Tensordot/Shape�
&decoder_output/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2(
&decoder_output/Tensordot/GatherV2/axis�
!decoder_output/Tensordot/GatherV2GatherV2'decoder_output/Tensordot/Shape:output:0&decoder_output/Tensordot/free:output:0/decoder_output/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2#
!decoder_output/Tensordot/GatherV2�
(decoder_output/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2*
(decoder_output/Tensordot/GatherV2_1/axis�
#decoder_output/Tensordot/GatherV2_1GatherV2'decoder_output/Tensordot/Shape:output:0&decoder_output/Tensordot/axes:output:01decoder_output/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2%
#decoder_output/Tensordot/GatherV2_1�
decoder_output/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2 
decoder_output/Tensordot/Const�
decoder_output/Tensordot/ProdProd*decoder_output/Tensordot/GatherV2:output:0'decoder_output/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
decoder_output/Tensordot/Prod�
 decoder_output/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 decoder_output/Tensordot/Const_1�
decoder_output/Tensordot/Prod_1Prod,decoder_output/Tensordot/GatherV2_1:output:0)decoder_output/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2!
decoder_output/Tensordot/Prod_1�
$decoder_output/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$decoder_output/Tensordot/concat/axis�
decoder_output/Tensordot/concatConcatV2&decoder_output/Tensordot/free:output:0&decoder_output/Tensordot/axes:output:0-decoder_output/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2!
decoder_output/Tensordot/concat�
decoder_output/Tensordot/stackPack&decoder_output/Tensordot/Prod:output:0(decoder_output/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2 
decoder_output/Tensordot/stack�
"decoder_output/Tensordot/transpose	Transposedecoder_hidden_layer1/Tanh:y:0(decoder_output/Tensordot/concat:output:0*
T0*+
_output_shapes
:��������� 2$
"decoder_output/Tensordot/transpose�
 decoder_output/Tensordot/ReshapeReshape&decoder_output/Tensordot/transpose:y:0'decoder_output/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2"
 decoder_output/Tensordot/Reshape�
)decoder_output/Tensordot/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2+
)decoder_output/Tensordot/transpose_1/perm�
$decoder_output/Tensordot/transpose_1	Transpose/decoder_output/Tensordot/ReadVariableOp:value:02decoder_output/Tensordot/transpose_1/perm:output:0*
T0*
_output_shapes
:	 �2&
$decoder_output/Tensordot/transpose_1�
(decoder_output/Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"    �   2*
(decoder_output/Tensordot/Reshape_1/shape�
"decoder_output/Tensordot/Reshape_1Reshape(decoder_output/Tensordot/transpose_1:y:01decoder_output/Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes
:	 �2$
"decoder_output/Tensordot/Reshape_1�
decoder_output/Tensordot/MatMulMatMul)decoder_output/Tensordot/Reshape:output:0+decoder_output/Tensordot/Reshape_1:output:0*
T0*(
_output_shapes
:����������2!
decoder_output/Tensordot/MatMul�
 decoder_output/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�2"
 decoder_output/Tensordot/Const_2�
&decoder_output/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2(
&decoder_output/Tensordot/concat_1/axis�
!decoder_output/Tensordot/concat_1ConcatV2*decoder_output/Tensordot/GatherV2:output:0)decoder_output/Tensordot/Const_2:output:0/decoder_output/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2#
!decoder_output/Tensordot/concat_1�
decoder_output/TensordotReshape)decoder_output/Tensordot/MatMul:product:0*decoder_output/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:����������2
decoder_output/Tensordot�
%decoder_output/BiasAdd/ReadVariableOpReadVariableOp.decoder_output_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02'
%decoder_output/BiasAdd/ReadVariableOp�
decoder_output/BiasAddBiasAdd!decoder_output/Tensordot:output:0-decoder_output/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������2
decoder_output/BiasAdd�
decoder_output/SigmoidSigmoiddecoder_output/BiasAdd:output:0*
T0*,
_output_shapes
:����������2
decoder_output/Sigmoid�
IdentityIdentitydecoder_output/Sigmoid:y:0-^decoder_hidden_layer1/BiasAdd/ReadVariableOp/^decoder_hidden_layer1/Tensordot/ReadVariableOp&^decoder_output/BiasAdd/ReadVariableOp(^decoder_output/Tensordot/ReadVariableOp-^encoder_hidden_layer1/BiasAdd/ReadVariableOp/^encoder_hidden_layer1/Tensordot/ReadVariableOp^latent/BiasAdd/ReadVariableOp ^latent/Tensordot/ReadVariableOp*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:����������::::::::2\
,decoder_hidden_layer1/BiasAdd/ReadVariableOp,decoder_hidden_layer1/BiasAdd/ReadVariableOp2`
.decoder_hidden_layer1/Tensordot/ReadVariableOp.decoder_hidden_layer1/Tensordot/ReadVariableOp2N
%decoder_output/BiasAdd/ReadVariableOp%decoder_output/BiasAdd/ReadVariableOp2R
'decoder_output/Tensordot/ReadVariableOp'decoder_output/Tensordot/ReadVariableOp2\
,encoder_hidden_layer1/BiasAdd/ReadVariableOp,encoder_hidden_layer1/BiasAdd/ReadVariableOp2`
.encoder_hidden_layer1/Tensordot/ReadVariableOp.encoder_hidden_layer1/Tensordot/ReadVariableOp2>
latent/BiasAdd/ReadVariableOplatent/BiasAdd/ReadVariableOp2B
latent/Tensordot/ReadVariableOplatent/Tensordot/ReadVariableOp:& "
 
_user_specified_nameinputs
�

�
+__inference_autoencoder_layer_call_fn_20102

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*
Tin
2	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*,
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_autoencoder_layer_call_and_return_conditional_losses_197872
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:����������::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
5__inference_encoder_hidden_layer1_layer_call_fn_20144

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*+
_output_shapes
:��������� *-
config_proto

CPU

GPU2*0J 8*Y
fTRR
P__inference_encoder_hidden_layer1_layer_call_and_return_conditional_losses_195692
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :����������::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
5__inference_decoder_hidden_layer1_layer_call_fn_20228

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*+
_output_shapes
:��������� *-
config_proto

CPU

GPU2*0J 8*Y
fTRR
P__inference_decoder_hidden_layer1_layer_call_and_return_conditional_losses_196632
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
.__inference_decoder_output_layer_call_fn_20270

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*,
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_decoder_output_layer_call_and_return_conditional_losses_197102
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*2
_input_shapes!
:��������� ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
&__inference_latent_layer_call_fn_20186

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*+
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_latent_layer_call_and_return_conditional_losses_196162
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������2

Identity"
identityIdentity:output:0*2
_input_shapes!
:��������� ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
F__inference_autoencoder_layer_call_and_return_conditional_losses_19739
encoder_input8
4encoder_hidden_layer1_statefulpartitionedcall_args_18
4encoder_hidden_layer1_statefulpartitionedcall_args_2)
%latent_statefulpartitionedcall_args_1)
%latent_statefulpartitionedcall_args_28
4decoder_hidden_layer1_statefulpartitionedcall_args_18
4decoder_hidden_layer1_statefulpartitionedcall_args_21
-decoder_output_statefulpartitionedcall_args_11
-decoder_output_statefulpartitionedcall_args_2
identity��-decoder_hidden_layer1/StatefulPartitionedCall�&decoder_output/StatefulPartitionedCall�-encoder_hidden_layer1/StatefulPartitionedCall�latent/StatefulPartitionedCall�
-encoder_hidden_layer1/StatefulPartitionedCallStatefulPartitionedCallencoder_input4encoder_hidden_layer1_statefulpartitionedcall_args_14encoder_hidden_layer1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*+
_output_shapes
:��������� *-
config_proto

CPU

GPU2*0J 8*Y
fTRR
P__inference_encoder_hidden_layer1_layer_call_and_return_conditional_losses_195692/
-encoder_hidden_layer1/StatefulPartitionedCall�
latent/StatefulPartitionedCallStatefulPartitionedCall6encoder_hidden_layer1/StatefulPartitionedCall:output:0%latent_statefulpartitionedcall_args_1%latent_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*+
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*J
fERC
A__inference_latent_layer_call_and_return_conditional_losses_196162 
latent/StatefulPartitionedCall�
-decoder_hidden_layer1/StatefulPartitionedCallStatefulPartitionedCall'latent/StatefulPartitionedCall:output:04decoder_hidden_layer1_statefulpartitionedcall_args_14decoder_hidden_layer1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*+
_output_shapes
:��������� *-
config_proto

CPU

GPU2*0J 8*Y
fTRR
P__inference_decoder_hidden_layer1_layer_call_and_return_conditional_losses_196632/
-decoder_hidden_layer1/StatefulPartitionedCall�
&decoder_output/StatefulPartitionedCallStatefulPartitionedCall6decoder_hidden_layer1/StatefulPartitionedCall:output:0-decoder_output_statefulpartitionedcall_args_1-decoder_output_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*,
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*R
fMRK
I__inference_decoder_output_layer_call_and_return_conditional_losses_197102(
&decoder_output/StatefulPartitionedCall�
IdentityIdentity/decoder_output/StatefulPartitionedCall:output:0.^decoder_hidden_layer1/StatefulPartitionedCall'^decoder_output/StatefulPartitionedCall.^encoder_hidden_layer1/StatefulPartitionedCall^latent/StatefulPartitionedCall*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:����������::::::::2^
-decoder_hidden_layer1/StatefulPartitionedCall-decoder_hidden_layer1/StatefulPartitionedCall2P
&decoder_output/StatefulPartitionedCall&decoder_output/StatefulPartitionedCall2^
-encoder_hidden_layer1/StatefulPartitionedCall-encoder_hidden_layer1/StatefulPartitionedCall2@
latent/StatefulPartitionedCalllatent/StatefulPartitionedCall:- )
'
_user_specified_nameencoder_input
�$
�
P__inference_encoder_hidden_layer1_layer_call_and_return_conditional_losses_20137

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp�
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	� *
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis�
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis�
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const�
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1�
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis�
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat�
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:����������2
Tensordot/transpose�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
Tensordot/Reshape�
Tensordot/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/transpose_1/perm�
Tensordot/transpose_1	Transpose Tensordot/ReadVariableOp:value:0#Tensordot/transpose_1/perm:output:0*
T0*
_output_shapes
:	� 2
Tensordot/transpose_1�
Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"�       2
Tensordot/Reshape_1/shape�
Tensordot/Reshape_1ReshapeTensordot/transpose_1:y:0"Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes
:	� 2
Tensordot/Reshape_1�
Tensordot/MatMulMatMulTensordot/Reshape:output:0Tensordot/Reshape_1:output:0*
T0*'
_output_shapes
:��������� 2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis�
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:��������� 2
	Tensordot�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:��������� 2	
BiasAdd\
TanhTanhBiasAdd:output:0*
T0*+
_output_shapes
:��������� 2
Tanh�
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:& "
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
L
encoder_input;
serving_default_encoder_input:0����������G
decoder_output5
StatefulPartitionedCall:0����������tensorflow/serving/predict:��
�&
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
	optimizer
regularization_losses
	variables
	trainable_variables

	keras_api

signatures
*Q&call_and_return_all_conditional_losses
R_default_save_signature
S__call__"�#
_tf_keras_sequential�#{"class_name": "Sequential", "name": "autoencoder", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "autoencoder", "layers": [{"class_name": "Dense", "config": {"name": "encoder_hidden_layer1", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "batch_input_shape": [null, 16, 202]}}, {"class_name": "Dense", "config": {"name": "latent", "trainable": true, "dtype": "float32", "units": 2, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "decoder_hidden_layer1", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "decoder_output", "trainable": true, "dtype": "float32", "units": 202, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 202}}}, "is_graph_network": true, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "autoencoder", "layers": [{"class_name": "Dense", "config": {"name": "encoder_hidden_layer1", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "batch_input_shape": [null, 16, 202]}}, {"class_name": "Dense", "config": {"name": "latent", "trainable": true, "dtype": "float32", "units": 2, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "decoder_hidden_layer1", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "decoder_output", "trainable": true, "dtype": "float32", "units": 202, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "binary_crossentropy", "metrics": [{"class_name": "BinaryCrossentropy", "config": {"name": "binary_crossentropy", "dtype": "float32", "from_logits": false, "label_smoothing": 0}}], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "RMSprop", "config": {"name": "RMSprop", "learning_rate": 0.0010000000474974513, "decay": 0.0, "rho": 0.8999999761581421, "momentum": 0.0, "epsilon": 1e-07, "centered": false}}}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "encoder_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": [null, 16, 202], "config": {"batch_input_shape": [null, 16, 202], "dtype": "float32", "sparse": false, "ragged": false, "name": "encoder_input"}}
�

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
*T&call_and_return_all_conditional_losses
U__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "encoder_hidden_layer1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "encoder_hidden_layer1", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 202}}}}
�

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
*V&call_and_return_all_conditional_losses
W__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "latent", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "latent", "trainable": true, "dtype": "float32", "units": 2, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}}
�

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
*X&call_and_return_all_conditional_losses
Y__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "decoder_hidden_layer1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "decoder_hidden_layer1", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2}}}}
�

kernel
bias
 regularization_losses
!	variables
"trainable_variables
#	keras_api
*Z&call_and_return_all_conditional_losses
[__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "decoder_output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "decoder_output", "trainable": true, "dtype": "float32", "units": 202, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}}
�
$iter
	%decay
&learning_rate
'momentum
(rho	rmsI	rmsJ	rmsK	rmsL	rmsM	rmsN	rmsO	rmsP"
	optimizer
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
�
)non_trainable_variables
*layer_regularization_losses
regularization_losses
	variables
	trainable_variables

+layers
,metrics
S__call__
R_default_save_signature
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses"
_generic_user_object
,
\serving_default"
signature_map
/:-	� 2encoder_hidden_layer1/kernel
(:& 2encoder_hidden_layer1/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
-layer_regularization_losses
regularization_losses
	variables

.layers
trainable_variables
/non_trainable_variables
0metrics
U__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses"
_generic_user_object
: 2latent/kernel
:2latent/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
1layer_regularization_losses
regularization_losses
	variables

2layers
trainable_variables
3non_trainable_variables
4metrics
W__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
_generic_user_object
.:, 2decoder_hidden_layer1/kernel
(:& 2decoder_hidden_layer1/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
5layer_regularization_losses
regularization_losses
	variables

6layers
trainable_variables
7non_trainable_variables
8metrics
Y__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
(:&	 �2decoder_output/kernel
": �2decoder_output/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
9layer_regularization_losses
 regularization_losses
!	variables

:layers
"trainable_variables
;non_trainable_variables
<metrics
[__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
:	 (2RMSprop/iter
: (2RMSprop/decay
: (2RMSprop/learning_rate
: (2RMSprop/momentum
: (2RMSprop/rho
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
'
=0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
	>total
	?count
@
_fn_kwargs
Aregularization_losses
B	variables
Ctrainable_variables
D	keras_api
*]&call_and_return_all_conditional_losses
^__call__"�
_tf_keras_layer�{"class_name": "BinaryCrossentropy", "name": "binary_crossentropy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "binary_crossentropy", "dtype": "float32", "from_logits": false, "label_smoothing": 0}}
:  (2total
:  (2count
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Elayer_regularization_losses
Aregularization_losses
B	variables

Flayers
Ctrainable_variables
Gnon_trainable_variables
Hmetrics
^__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
9:7	� 2(RMSprop/encoder_hidden_layer1/kernel/rms
2:0 2&RMSprop/encoder_hidden_layer1/bias/rms
):' 2RMSprop/latent/kernel/rms
#:!2RMSprop/latent/bias/rms
8:6 2(RMSprop/decoder_hidden_layer1/kernel/rms
2:0 2&RMSprop/decoder_hidden_layer1/bias/rms
2:0	 �2!RMSprop/decoder_output/kernel/rms
,:*�2RMSprop/decoder_output/bias/rms
�2�
F__inference_autoencoder_layer_call_and_return_conditional_losses_19948
F__inference_autoencoder_layer_call_and_return_conditional_losses_20076
F__inference_autoencoder_layer_call_and_return_conditional_losses_19723
F__inference_autoencoder_layer_call_and_return_conditional_losses_19739�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
 __inference__wrapped_model_19530�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *1�.
,�)
encoder_input����������
�2�
+__inference_autoencoder_layer_call_fn_19769
+__inference_autoencoder_layer_call_fn_20089
+__inference_autoencoder_layer_call_fn_19798
+__inference_autoencoder_layer_call_fn_20102�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
P__inference_encoder_hidden_layer1_layer_call_and_return_conditional_losses_20137�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
5__inference_encoder_hidden_layer1_layer_call_fn_20144�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
A__inference_latent_layer_call_and_return_conditional_losses_20179�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
&__inference_latent_layer_call_fn_20186�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
P__inference_decoder_hidden_layer1_layer_call_and_return_conditional_losses_20221�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
5__inference_decoder_hidden_layer1_layer_call_fn_20228�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
I__inference_decoder_output_layer_call_and_return_conditional_losses_20263�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
.__inference_decoder_output_layer_call_fn_20270�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
8B6
#__inference_signature_wrapper_19820encoder_input
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 �
 __inference__wrapped_model_19530�;�8
1�.
,�)
encoder_input����������
� "D�A
?
decoder_output-�*
decoder_output�����������
F__inference_autoencoder_layer_call_and_return_conditional_losses_19723{C�@
9�6
,�)
encoder_input����������
p

 
� "*�'
 �
0����������
� �
F__inference_autoencoder_layer_call_and_return_conditional_losses_19739{C�@
9�6
,�)
encoder_input����������
p 

 
� "*�'
 �
0����������
� �
F__inference_autoencoder_layer_call_and_return_conditional_losses_19948t<�9
2�/
%�"
inputs����������
p

 
� "*�'
 �
0����������
� �
F__inference_autoencoder_layer_call_and_return_conditional_losses_20076t<�9
2�/
%�"
inputs����������
p 

 
� "*�'
 �
0����������
� �
+__inference_autoencoder_layer_call_fn_19769nC�@
9�6
,�)
encoder_input����������
p

 
� "������������
+__inference_autoencoder_layer_call_fn_19798nC�@
9�6
,�)
encoder_input����������
p 

 
� "������������
+__inference_autoencoder_layer_call_fn_20089g<�9
2�/
%�"
inputs����������
p

 
� "������������
+__inference_autoencoder_layer_call_fn_20102g<�9
2�/
%�"
inputs����������
p 

 
� "������������
P__inference_decoder_hidden_layer1_layer_call_and_return_conditional_losses_20221d3�0
)�&
$�!
inputs���������
� ")�&
�
0��������� 
� �
5__inference_decoder_hidden_layer1_layer_call_fn_20228W3�0
)�&
$�!
inputs���������
� "���������� �
I__inference_decoder_output_layer_call_and_return_conditional_losses_20263e3�0
)�&
$�!
inputs��������� 
� "*�'
 �
0����������
� �
.__inference_decoder_output_layer_call_fn_20270X3�0
)�&
$�!
inputs��������� 
� "������������
P__inference_encoder_hidden_layer1_layer_call_and_return_conditional_losses_20137e4�1
*�'
%�"
inputs����������
� ")�&
�
0��������� 
� �
5__inference_encoder_hidden_layer1_layer_call_fn_20144X4�1
*�'
%�"
inputs����������
� "���������� �
A__inference_latent_layer_call_and_return_conditional_losses_20179d3�0
)�&
$�!
inputs��������� 
� ")�&
�
0���������
� �
&__inference_latent_layer_call_fn_20186W3�0
)�&
$�!
inputs��������� 
� "�����������
#__inference_signature_wrapper_19820�L�I
� 
B�?
=
encoder_input,�)
encoder_input����������"D�A
?
decoder_output-�*
decoder_output����������