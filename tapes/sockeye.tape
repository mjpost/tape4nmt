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

# tasks related to subword processing
import "source_factors.tape"

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


task prepare_data : sockeye
    < train_src_in=$prepared_data_src[DataSection:train]
    < train_trg_in=$prepared_data_trg[DataSection:train]
    < factor_files=(SourceFactors: no="/dev/null" yes=$factor_files@compute_source_factors[DataSection:train])
    > data
    :: train_maxlen=@
    :: source_factors=@
    :: seed=@
    :: .submitter=$submitter
    :: .resource_flags=$resource_flags_16g
    :: .pyenv=@
    :: .action_flags=@ {

  export PYTHONPATH=${sockeye}

  source_factors_flag=""
  if [[ $factor_files != "/dev/null" ]]; then
    source_factors_flag="--source-factors $(cat $factor_files)"
  fi

  conda deactivate
  conda activate sockeye
  module unload cuda100/toolkit
  module load cuda92/toolkit

  python3 -m sockeye.prepare_data \
      --source $train_src_in \
      $source_factors_flag \
      --target $train_trg_in \
      --shared-vocab \
      --word-min-count 2:2 \
      --bucket-width 10 \
      --max-seq-len $train_maxlen \
      --num-samples-per-shard 10000000 \
      --seed $seed \
      --output data
}

task sockeye_train : sockeye
    < prepared_data=$data@prepare_data
    < dev_src=$prepared_data_src[DataSection:dev]
    < dev_trg=$prepared_data_trg[DataSection:dev]
    < dev_factor_files=(SourceFactors: no="/dev/null" yes=$factor_files@compute_source_factors[DataSection:dev])
    > model
    :: use_source_factors=(SourceFactors: no yes)
    :: train_batch_type=@
    :: train_batch_size=@
    :: train_max_checkpoints_not_improved=@
    :: train_checkpoint_freq=@
    :: train_num_decode_and_eval=@
    :: encoder_type=@
    :: decoder_type=@
    :: num_layers=@
    :: num_embed=@
    :: transformer_model_size=@
    :: transformer_attention_heads=@
    :: transformer_feed_forward_num_hidden=@
    :: initial_learning_rate=@
    :: label_smoothing=@
    :: source_factors=@
    :: source_factors_num_embed=@
    :: source_factors_combine=@
    :: use_cpu=@
    :: num_devices=@
    :: seed=$train_seed
    :: .submitter=$submitter
    :: .resource_flags=$resource_flags_train
    :: .pyenv=@
    :: .action_flags=$action_flags_success {

  if [[ $use_cpu == "yes" ]]; then
    device="--use-cpu"
  else
    device="--device-ids -$num_devices"
  fi

  dev_factors_flag=""
  source_factors_flag=""
  if [[ $use_source_factors == "yes" ]]; then
    dev_factors_flag="--validation-source-factors $(cat $dev_factor_files)"
    source_factors_flag="--source-factors-num-embed $source_factors_num_embed --source-factors-combine $source_factors_combine"
  fi

  export PYTHONPATH=${sockeye}

  python3 -m sockeye.train \
    -o $model \
    $device \
    --seed=$seed \
    --disable-device-locking \
    --prepared-data $prepared_data \
    --num-embed=$num_embed \
    $source_factors_flag \
    --validation-source $dev_src \
    --validation-target $dev_trg \
    $dev_factors_flag \
    --encoder=$encoder_type \
    --decoder=$decoder_type \
    --num-layers=$num_layers \
    --transformer-model-size=$transformer_model_size \
    --transformer-attention-heads=$transformer_attention_heads \
    --transformer-feed-forward-num-hidden=$transformer_feed_forward_num_hidden \
    --transformer-positional-embedding-type=fixed \
    --transformer-preprocess=n \
    --transformer-postprocess=dr \
    --transformer-dropout-attention=0.1 \
    --transformer-dropout-act=0.1 \
    --transformer-dropout-prepost=0.1 \
    --weight-tying \
    --weight-tying-type=src_trg_softmax \
    --weight-init=xavier \
    --weight-init-scale=3.0 \
    --weight-init-xavier-factor-type=avg \
    --optimizer=adam \
    --optimized-metric=perplexity \
    --label-smoothing=$label_smoothing \
    --gradient-clipping-threshold=-1 \
    --initial-learning-rate=$initial_learning_rate \
    --learning-rate-reduce-num-not-improved=8 \
    --learning-rate-reduce-factor=0.9 \
    --learning-rate-scheduler-type=plateau-reduce \
    --learning-rate-decay-optimizer-states-reset=best \
    --learning-rate-decay-param-reset \
    --max-num-checkpoint-not-improved $train_max_checkpoints_not_improved \
    --batch-type=$train_batch_type \
    --batch-size=$train_batch_size \
    --checkpoint-frequency=$train_checkpoint_freq \
    --decode-and-evaluate=$train_num_decode_and_eval \
    --decode-and-evaluate-use-cpu \
    --keep-last-params=10
}

# Takes all the data and builds a packaged model directory.
# Ideally this should be used for evaluation.
task package_model : sockeye mosesdecoder sockeye_scripts fast_align
  < subword_model_src=(SubwordMethod: sentencepiece=$src_model@train_sentencepiece bpe=$src_model@train_bpe none="/dev/null")
  < subword_model_trg=(SubwordMethod: sentencepiece=$trg_model@train_sentencepiece bpe=$trg_model@train_bpe none="/dev/null")
  < casing_model=(Casing: actual="/dev/null" lower="/dev/null" lower_source="/dev/null" true=$out@train_truecaser)
  < model=$model@sockeye_train
  < observed_masks=(DoMasking: no="/dev/null" yes=$masks@mask[DataSection:train])
  < forward_log=(MaskIndex: yes="/dev/null" no=$forward_log@fast_align)
  < reverse_log=(MaskIndex: yes="/dev/null" no=$reverse_log@fast_align)
  < forward_lexicon=(MaskIndex: yes="/dev/null" no=$forward_lexicon@fast_align)
  < reverse_lexicon=(MaskIndex: yes="/dev/null" no=$reverse_lexicon@fast_align)
  > bundle="bundle"
  :: casing=(Casing: actual lower lower_source true)
  :: tokenizer=(Tokenizer: none default cjk)
  :: mask=(DoMasking: no="no" yes="yes")
  :: masking_pattern_files=@
  :: masking_dict_files=@
  :: masking_add_index=@
  :: masking_unmask=@
  :: tokenize_protect=@
  :: subword_method=(SubwordMethod: sentencepiece bpe none)
  :: use_factors=(SourceFactors: no yes)
  :: source_factors=@
  :: SRC=@
  :: TRG=@
  :: .submitter=shell {

  mkdir bundle
  cd bundle
  cp -a ${sockeye} sockeye
  cp -a ${sockeye_scripts} sockeye_scripts

  # fast_align
  mkdir bin
  cp ${fast_align}/build/{fast_align,atools} bin

  ## Pre-processing and post-processing
  rm -f pre.sh
  echo "#!/usr/bin/env bash" >> pre.sh
  echo >> pre.sh
  echo "lang=\${1:-$SRC}" >> pre.sh
  echo "rundir=\$(dirname \$0)" >> pre.sh

  rm -f post.sh
  echo "#!/usr/bin/env bash" >> post.sh
  echo >> post.sh
  echo "lang=\${1:-$TRG}" >> post.sh
  echo "rundir=\$(dirname \$0)" >> post.sh
  echo >> post.sh
  echo "export PATH+=:\$rundir/bin"   >> post.sh
  echo >> post.sh

  # Creates a JSON object from the plain input
  pre_cmd="\$rundir/sockeye_scripts/preparation/wrap_in_json -r raw_text"
  # copy the "translation" field to "text" since Sockeye doesn't overwrite that
  post_cmd="\$rundir/sockeye_scripts/preparation/wrap_in_json translation text"

  output_attention_flag=""

  if [[ $mask == "yes" ]]; then
    pre_cmd+=" | python3 \$rundir/sockeye_scripts/masking/mask_terms.py --json"

    [[ ! -d data ]] && mkdir data

    if [[ $masking_pattern_files != "/dev/null" ]]; then
      cp $masking_pattern_files data/masking_patterns.txt
      pre_cmd+=" --pattern-files \$rundir/data/masking_patterns.txt"
    fi
    if [[ $masking_dict_files != "/dev/null" ]]; then
      cp $masking_dict_files data/masking_dict.txt
      pre_cmd+=" --dict-files \$rundir/data/masking_dict.txt"
    fi
    if [[ $masking_add_index == "yes" ]] ; then
      pre_cmd+=" --add-index"
    fi

    if [[ $masking_unmask == "attention" ]]; then
      post_cmd+=" | python3 \$rundir/sockeye_scripts/masking/bipar.py"
      output_attention_flag=" --output-attention"
    elif [[ $masking_unmask == "alignment" ]]; then
      post_cmd+=" | python3 \$rundir/sockeye_scripts/preparation/prepare.py --input-field text --undo --subword-type $subword_method"
      post_cmd+=" | python3 \$rundir/sockeye_scripts/masking/add_alignment.py $forward_lexicon $forward_log $reverse_lexicon $reverse_log"
      post_cmd+=" | python3 \$rundir/sockeye_scripts/masking/bipar.py"
      post_cmd+=" | \$rundir/sockeye_scripts/preparation/prepare.py --input-field text --undo --subword-type $subword_method"
    else
      # Align using the indices
      post_cmd+=" | python3 \$rundir/sockeye_scripts/masking/mask_terms.py --json --unmask"
      post_cmd+=" | \$rundir/sockeye_scripts/preparation/prepare.py --input-field text --undo --subword-type $subword_method"
    fi
  else
    # just remove the subword processing
    post_cmd+=" | \$rundir/sockeye_scripts/preparation/prepare.py --input-field text --undo --subword-type $subword_method"
  fi

  if [[ $tokenizer == "cjk" ]]; then
    mkdir tokenizer
    cp ${sockeye_scripts}/cjk/segment-cjk.py tokenizer/
    pre_cmd+=" | \$rundir/tokenizer/segment-cjk.py --json"
  fi

  prepare_casing_arg=""
  if [[ $casing == "lower_source" ]]; then
    prepare_casing_arg="--casing lower"
  elif [[ $casing == "true" ]]; then
    # cp -a ${mosesdecoder}/scripts/recaser .
    # cp ${casing_model} truecase.model
    # pre_cmd+=" | \$rundir/sockeye_scripts/preparation/wrap_in_json text recased_text \"\$rundir/recaser/truecase.perl -b -model \$rundir/truecase.model\""
    prepare_casing_arg="--casing true"
  fi

  prepare_subword_arg=""
  if [[ $subword_model_src != "/dev/null" ]]; then
    cp $subword_model_src subword.src.model
    prepare_subword_arg="--subword-type $subword_method --subword-model \$rundir/subword.src.model"

    if [[ $mask == "yes" && $subword_method == "bpe" ]]; then
      prepare_subword_arg+=" --subword-glossary '__[A-Zz-z0-9]+(_\d+)?__'"
    fi
  fi

  pre_cmd+=" | \$rundir/sockeye_scripts/preparation/prepare.py $prepare_casing_arg $prepare_subword_arg"

  if [[ $use_factors == "yes" ]]; then
    pre_cmd+=" | PYTHONPATH=\$rundir/sockeye_scripts python3 -m source_factors.compute --json $source_factors"
  fi  

  if [[ $tokenizer == "default" ]]; then
    post_cmd+=" | \$rundir/sockeye_scripts/preparation/wrap_in_json text detok_translation \"\$rundir/tokenizer/detokenizer.perl -q -b -l $TRG\""
  fi

  echo $pre_cmd >> pre.sh
  echo $post_cmd >> post.sh

  ## Translate
  cp -a ${model} model
  cat > translate.sh <<EOF
#!/usr/bin/env bash

bundledir=\$(dirname \$0)

\$bundledir/pre.sh ${SRC} \\
  | PYTHONPATH=\$bundledir/sockeye python3 -m sockeye.translate \\
    -q \\
    -m \$bundledir/model \\
    --json-input --output-type json \\
    $output_attention_flag \\
    "\$@" \\
  | \$bundledir/post.sh
EOF

  # permissions
  chmod 755 pre.sh post.sh translate.sh
}

task decode : sockeye tools
    < in=$raw_data_test_src
    < bundled_model=$bundle@package_model
    > out="out"
    > log="out.log"
    < forward_log=(MaskIndex: yes="/dev/null" no=$forward_log@fast_align)
    < reverse_log=(MaskIndex: yes="/dev/null" no=$reverse_log@fast_align)
    < forward_lexicon=(MaskIndex: yes="/dev/null" no=$forward_lexicon@fast_align)
    < reverse_lexicon=(MaskIndex: yes="/dev/null" no=$reverse_lexicon@fast_align)
    :: do_masking=(DoMasking: no="no" yes="yes")
    :: masking_add_index=@
    :: num_devices=@
    :: test_max_sent_length=@
    :: test_beam_size=@
    :: test_batch_size=@
    :: use_cpu=@
    :: .submitter=$submitter
    :: .action_flags=@
    :: .resource_flags=$resource_flags_decode
    :: .pyenv=@ {

  if [[ $use_cpu == "yes" ]]; then
    device_flag="--use-cpu"
  else
    device_flag="--device-ids -1"
  fi

  ln -sf $in corpus.in
  ln -sf ${bundled_model} bundle

  cat $in | ${bundled_model}/translate.sh \
    $device_flag \
    --disable-device-locking \
    --beam-size $test_beam_size \
    --batch-size $test_batch_size \
    --max-input-len $test_max_sent_length \
    | tee ${out}.json | jq -r .text > $out 2> $log

  # TEMPORARY HACK
  # if [[ $do_masking == "yes" && $masking_add_index == "no" ]]; then
  #   cat corpus | python3 ${fast_align}/src/force_align.py $forward_lexicon $forward_log $reverse_lexicon $reverse_log grow-diag-final-and > alignments
  #   cat ${out}.json | python3 ${sockeye_scripts}/masking/replace_masks.py -a alignments > t
  # fi
}

task decode_straight : sockeye tools
  < in=$prepared_data_src[DataSection:test]
  < model=$model@sockeye_train
  > out="out"
  :: num_devices=@
  :: test_max_sent_length=@
  :: test_beam_size=@
  :: test_batch_size=@
  :: use_cpu=@
  :: .submitter=$submitter
  :: .action_flags=@
  :: .resource_flags=$resource_flags_decode
  :: .pyenv=@ {

  if [[ $use_cpu == "yes" ]]; then
    device_flag="--use-cpu"
  else
    device_flag="--device-ids -1"
  fi

  ln -sf $in corpus.in

  export PYTHONPATH=${sockeye}
  cat $in | python3 -m sockeye.translate \
    -m $model \
    $device_flag \
    --disable-device-locking \
    --beam-size $test_beam_size \
    --batch-size $test_batch_size \
    --max-input-len $test_max_sent_length \
    > $out
}
