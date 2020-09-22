global {
  prepared_data_src=(SubwordMethod:
    sentencepiece=$src@apply_sentencepiece
    bpe=$src@apply_bpe
    none=$recased_data_src
  )
  prepared_data_trg=(SubwordMethod:
    sentencepiece=$trg@apply_sentencepiece
    bpe=$trg@apply_bpe
    none=$recased_data_trg
  )
}

task train_bpe : subword_nmt
  < src_in=$recased_data_src[DataSection:train]
  < trg_in=$recased_data_trg[DataSection:train]
  > src_model="bpe.model.src"
  > trg_model="bpe.model.trg"
  :: subword_joint_model=@
  :: bpe_operations=$subword_vocab_size
  :: SRC=@
  :: TRG=@
  :: .submitter=$submitter .action_flags=$action_flags .resource_flags=$resource_flags_16g {

  if [[ $subword_joint_model == 'yes' ]]; then
    subword-nmt learn-joint-bpe-and-vocab -i $src_in $trg_in -s $bpe_operations -o $src_model --write-vocabulary bpe.vocab.$SRC bpe.vocab.$TRG -v 2> log
    ln -sf $src_model $trg_model
  else
    subword-nmt learn-bpe -i $src_in -s $bpe_operations -o $src_model -v 2> log.src &
    subword-nmt learn-bpe -i $trg_in -s $bpe_operations -o $trg_model -v 2> log.trg &
    wait
  fi
}

task apply_bpe : subword_nmt
  < src_in=$recased_data_src
  < trg_in=$recased_data_trg
  < src_model=$src_model@train_bpe
  < trg_model=$trg_model@train_bpe
  > src
  > trg
  :: .submitter=$submitter .action_flags=$action_flags .resource_flags=$resource_flags_16cpu {

  glossary_flag=""

  echo "GLOSSARY FLAG: $glossary_flag"

  # cat $src_in | parallel -j 10 -k --block-size 10k --pipe "subword-nmt apply-bpe --codes $model $glossary_flag" > $src
  # cat $trg_in | parallel -j 10 -k --block-size 10k --pipe "subword-nmt apply-bpe --codes $model $glossary_flag" > $trg

  cat $src_in | subword-nmt apply-bpe --codes $src_model $glossary_flag > $src &
  cat $trg_in | subword-nmt apply-bpe --codes $trg_model $glossary_flag > $trg &
  wait
}

task train_sentencepiece : sentencepiece
  < src_in=$recased_data_src[DataSection:train]
  < trg_in=$recased_data_trg[DataSection:train]
  > src_model="sp.src.model"
  > trg_model="sp.trg.model"
  > src_vocab="sp.src.vocab"
  > trg_vocab="sp.trg.vocab"
  :: subword_joint_model=@
  :: sentencepiece_vocab_size=$subword_vocab_size
  :: sentencepiece_model_type=@
  :: subword_coverage_src=@
  :: subword_coverage_trg=@
  :: .submitter=$submitter
  :: .resource_flags=$resource_flags_16cpu
  :: .action_flags=@ {

  glossary_flag=""

  if [[ $subword_joint_model == "yes" ]]; then
    ${sentencepiece}/build/src/spm_train \
      --input $src_in,$trg_in \
      --model_prefix sp \
      --vocab_size $sentencepiece_vocab_size \
      $glossary_flag \
      --character_coverage $subword_coverage_src \
      --model_type $sentencepiece_model_type \
      --input_sentence_size=10000000 \
      --shuffle_input_sentence=true

    ln -sf sp.model $src_model
    ln -sf sp.model $trg_model
    ln -sf sp.vocab $src_vocab
    ln -sf sp.vocab $trg_vocab

  else
    vocab_size_src=$(echo $sentencepiece_vocab_size | cut -d: -f1)
    vocab_size_trg=$(echo $sentencepiece_vocab_size | cut -d: -f2)

    ${sentencepiece}/build/src/spm_train --input $src_in \
      --model_prefix sp.src \
      --vocab_size $vocab_size_src \
      --character_coverage $subword_coverage_src \
      --input_sentence_size 10000000 \
      --shuffle_input_sentence=true \
      --model_type $sentencepiece_model_type &
    ${sentencepiece}/build/src/spm_train --input $trg_in \
      --model_prefix sp.trg \
      --vocab_size $vocab_size_trg $glossary_flag \
      --character_coverage $subword_coverage_trg \
      --input_sentence_size 10000000 \
      --shuffle_input_sentence=true \
      --model_type $sentencepiece_model_type
    wait
  fi
}

task apply_sentencepiece : sentencepiece
  < src_in=$recased_data_src
  < trg_in=$recased_data_trg
  < src_model=$src_model@train_sentencepiece
  < trg_model=$trg_model@train_sentencepiece
  < src_vocab=$src_vocab@train_sentencepiece
  < trg_vocab=$trg_vocab@train_sentencepiece
  > src
  > trg
  :: sentencepiece_alpha=@ {

  # Don't add this! The vocabulary is not the vocabulary:  --vocabulary=$src_vocab \
  cat $src_in \
  | ${sentencepiece}/build/src/spm_encode \
    --model $src_model \
    --output_format sample_piece \
    --alpha $sentencepiece_alpha \
  > $src &

  # Don't add this! The vocabulary is not the vocabulary:  --vocabulary=$trg_vocab \
  cat $trg_in \
  | ${sentencepiece}/build/src/spm_encode \
    --model $trg_model \
    --output_format sample_piece \
    --alpha $sentencepiece_alpha \
   > $trg &
  wait
}
