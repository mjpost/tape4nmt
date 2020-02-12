import "packages.tape"
import "submitters.tape"
import "versioners.tape"

# ==== pipeline starts here ====

# download all the data to the local directory
import "download.tape"

# perform masking on reserved symbols to improve consistency 
import "mask.tape"

# fast align for unmasking
import "fast_align.tape"

# tasks related to tokenize
import "tokenize.tape"

# tasks related to casing
import "recase.tape"

# tasks related to subword processing
import "subword.tape"

# TODO: source factors for fairseq?
# import "source_factors.tape"

# sacrebleu
import "bleu.tape"

# ==== pipeline ends here ====

# Nuts and bolts:
global {
  ducttape_experimental_packages=true
  ducttape_experimental_submitters=true
  ducttape_experimental_imports=true
  ducttape_experimental_multiproc=true
}

task binarize_data : fairseq
    < train_src_in=$prepared_data_src[DataSection:train]
    < train_trg_in=$prepared_data_trg[DataSection:train]
    < dev_src_in=$prepared_data_src[DataSection:dev]
    < dev_trg_in=$prepared_data_trg[DataSection:dev]
    > data
    :: SRC=@
    :: TRG=@
    :: train_shared_vocab=@
    :: .submitter=$submitter
    :: .resource_flags=$resource_flags_16g
    :: .pyenv=@
    :: .action_flags=@ {

  ln -s $train_src_in train.$SRC
  ln -s $train_trg_in train.$TRG
  ln -s $dev_src_in dev.$SRC
  ln -s $dev_trg_in dev.$TRG

  join=""
  if [[ $train_shared_vocab == "yes" ]]; then
    join=" --joined-dictionary"
  fi

  fairseq-preprocess \
    --source-lang $SRC --target-lang $TRG \
    --trainpref train \
    --validpref dev \
    --workers 30 \
    $join \
    --destdir $data
}

task fairseq_train : fairseq
    < in=$data@binarize_data
    > model
    :: SRC=@
    :: TRG=@
    :: train_arch=@
    :: train_max_tokens=@
    :: train_optim=@
    :: train_dropout=@
    :: train_adam_betas=@
    :: train_num_encoder_layers=@
    :: train_num_decoder_layers=@
    :: train_model_size=@
    :: train_ff_size=@
    :: train_num_heads=@
    :: train_lr=@
    :: train_lr_min=@
    :: train_lr_scheduler=@
    :: train_weight_decay=@
    :: train_warmup_init_lr=@
    :: train_warmup_updates=@
    :: train_criterion=@
    :: train_label_smoothing=@
    :: train_clip_norm=@
    :: train_shared_embeddings=@
    :: train_skip_invalid_size_inputs_valid_test=@
    :: train_update_freq=@
    :: train_patience=@
    :: train_max_epoch=@
    :: .submitter=sge
    :: .resource_flags=$resource_flags_train
    :: .pyenv=@
    :: .action_flags=@ {

  fairseq-train \
    $in \
    --patience $train_patience \
    --save-dir $model \
    --arch $train_arch \
    --max-tokens $train_max_tokens \
    --update-freq $train_update_freq \
    --save-interval-updates 100 \
    --keep-interval-updates 400 \
    --max-epoch $train_max_epoch \
    --num-workers 5 \
    --fp16 \
    --memory-efficient-fp16 \
    --num-workers 0 \
    --source-lang $SRC \
    --target-lang $TRG \
    --seed 2 \
    --arch transformer \
    --encoder-layers $train_num_encoder_layers \
    --decoder-layers $train_num_decoder_layers \
    --encoder-embed-dim $train_model_size \
    --decoder-embed-dim $train_model_size \
    --encoder-ffn-embed-dim $train_ff_size \
    --decoder-ffn-embed-dim $train_ff_size  \
    --encoder-attention-heads $train_num_heads \
    --decoder-attention-heads $train_num_heads \
    $train_shared_embeddings \
    --dropout $train_dropout \
    --attention-dropout $train_dropout \
    --relu-dropout $train_dropout \
    --weight-decay $train_weight_decay \
    --criterion $train_criterion \
    --label-smoothing $train_label_smoothing \
    --optimizer $train_optim \
    --adam-betas "$train_adam_betas" \
    --clip-norm $train_clip_norm \
    --lr $train_lr \
    --min-lr $train_lr_min \
    --lr-scheduler $train_lr_scheduler \
    --warmup-init-lr $train_warmup_init_lr \
    --warmup-updates $train_warmup_updates \
    --clip-norm $train_clip_norm \
    --ddp-backend=no_c10d \
    --no-last-checkpoints \
    --no-epoch-checkpoints \
    --log-format json --log-interval 1  > log
}


task package_model : fairseq bin sockeye_scripts
  < subword_model=(SubwordMethod: sentencepiece=$model@train_sentencepiece bpe=$src_model@train_bpe none="/dev/null")
  < data=$data@binarize_data
  < model=$model@fairseq_train
  > package="package"
  :: SRC=@
  :: TRG=@
  :: .submitter=$submitter
  :: .action_flags=@
  :: .resource_flags=$resource_flags_decode
  :: .pyenv=@ {

  mkdir $package
  cd $package

  ln $model/checkpoint_best.pt checkpoint_best.pt
  ln $subword_model subword.model

  ln -s $data/dict.$SRC.txt dict.src.txt
  ln -s $data/dict.$TRG.txt dict.trg.txt

  cat > preprocess.sh << EOF
#!/usr/bin/env bash
set -eux
spm_encode --model $(dirname \$0)/subword.model
EOF

  cat > translate.sh << EOF
#!/bin/bash
#
# Usage: 
#
#   translate.sh source.txt output.txt [args]
#
# Translates from file source.txt into language LANG, optionally 
# where
#
# - source.txt is the source file
# - args is optional arguments passed to fairseq-generate

#\$ -S /bin/bash -cwd -j y -o logs/
#\$ -l gpu=1,num_proc=1,h_rt=0:59:00 -q gpu_short.q

set -eu

module load cuda10.1/toolkit cudnn

# path to directory containing 'dict.src.txt' and 'checkpoint_best.pt'
model=\$(abspath \$0)

# path to source file
source=\$1

# path to write to
output=\$2

shift
shift

if [[ -s \$output ]]; then
    echo "$output already exists (\$(cat \$output | wc -l) lines), quitting."
    exit
fi

[[ ! -d \$(dirname \$output) ]] && mkdir -p \$(dirname \$output)

tmpdir=\$(mktemp -d --tmpdir=/expscratch/\$USER)
echo "TMPDIR=\$tmpdir"
echo "HOST=\$(hostname)"
env | grep CUDA
env | grep SGE

if [[ \$source == sacrebleu://* ]]; then
    # format: "sacrebleu://test-set/source-target
    testset=\$(echo \$source | cut -d/ -f3)
    langpair=\$(echo \$source | cut -d/ -f4)
    sacrebleu -t \$testset -l \$langpair --echo src | \$model/preprocess.sh > \$tmpdir/test.src
else
    cat \$source | \$model/preprocess.sh > \$tmpdir/test.src
fi

# dummy file
cp \$tmpdir/test.src \$tmpdir/test.trg

fairseq-preprocess --source-lang src --target-lang trg --testpref \$tmpdir/test --destdir \$tmpdir --tgtdict \$model/dict.trg.txt --srcdict \$model/dict.src.txt > \$tmpdir/preprocess.log

fairseq-generate \$tmpdir --path \$model/checkpoint_best.pt --gen-subset test --max-sentences 20 "\$@" > \$tmpdir/out
mv \$tmpdir/out \$output

rm -rf \$tmpdir
EOF

  chmod 755 preprocess.sh translate.sh
}

# the target input here is used to compute naive acc and ppl,
# that's why we need post-bpe target input
task decode : fairseq bin
    < in=$prepared_data_src[DataSection:test]
    < binarized_data_dir=$out@binarize_data
    < model=$model@fairseq_train
    > out
    :: test_max_sent_length=@
    :: test_beam_size=@
    :: test_batch_size=@
    :: SRC=@
    :: TRG=@
    :: .submitter=$submitter
    :: .action_flags=@
    :: .resource_flags=$resource_flags_decode
    :: .pyenv=@ {

  cmd="python $fairseq/interactive.py --path $model/checkpoint_best.pt $binarized_data_dir --beam $test_beam_size --source-lang $SRC --target-lang $TRG"

  echo $cmd
  $cmd < $in > decode.log
  cat decode.log | grep ^H | cut -f3 | ${bin}/debpe > $out
}
