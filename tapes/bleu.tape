task multi_bleu : mosesdecoder
    < in=$out@decode
    < ref=$raw_data_test_trg
    > bleu
    > bleu_c
    :: .submitter=$submitter .action_flags=$action_flags .resource_flags=$resource_flags {

  refn=`ls $ref* | wc -l`
  if [ $refn -gt 1 ] ; then
    mv $ref ${ref}.tmp # get rid of dummy single reference
  fi

  $mosesdecoder/scripts/generic/multi-bleu-detok.perl $ref < $in > $bleu_c
  $mosesdecoder/scripts/generic/multi-bleu-detok.perl -lc $ref < $in > $bleu

  if [ $refn -gt 1 ] ; then
    mv ${ref}.tmp $ref
  fi
}

task sacrebleu
    < sys_in=$out@decode
    < ref_in=$raw_data_test_trg
    > bleu
    > signature
    :: sacrebleu_args=@
    :: TRG=@
    :: .submitter=shell
    :: .pyenv=@ {

  # probably should make this a package
  kytea=~/code/kytea

  sys=$sys_in
  ref=$ref_in  
  if [[ $TRG == "ja" ]]; then
    cat $sys_in | ${kytea}/src/bin/kytea -model ${kytea}/data/model.bin -out tok > sys
    sys=sys
    cat $ref_in | ${kytea}/src/bin/kytea -model ${kytea}/data/model.bin -out tok > ref
    ref=ref
  fi

  cat $sys | sacrebleu $sacrebleu_args $ref | cut -d' ' -f1 > $signature
  cat $sys | sacrebleu $sacrebleu_args $ref -b > $bleu
}

task sacrebleu_straight : sentencepiece mosesdecoder
  < sys=$out@decode_straight
  < ref=$raw_data_test_trg
  < sp_model=$trg_model@train_sentencepiece
  > bleu
  > signature
  :: tokenizer=(Tokenizer: none default cjk)
  :: sacrebleu_args=@
  :: TRG=@
  :: .submitter=shell
  :: .pyenv=@ {

  ln $sys sys
  sys="sys"

  cat $sys | ${sentencepiece}/build/src/spm_decode --model $sp_model > sys.debpe
  sys="sys.debpe"

  if [[ $tokenizer == "default" ]]; then
    cat $sys | ${mosesdecoder}/scripts/tokenizer/detokenizer.perl -l $lang -b -q > sys.detok
    sys="sys.detok"
  fi

  cat $sys | sacrebleu $sacrebleu_args $ref | cut -d' ' -f1 > $signature
  cat $sys | sacrebleu $sacrebleu_args $ref -b > $bleu
}
