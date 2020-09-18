global {
  tokenized_data_src=(Tokenizer:
    none=$raw_data_src
    default=$src@tokenize
    cjk=$src@tokenize
  )
  tokenized_data_trg=(Tokenizer:
    none=$raw_data_trg
    default=$trg@tokenize
    cjk=$trg@tokenize
  )
}

task tokenize : mosesdecoder tools sockeye_scripts
  < src_in=$raw_data_src
  < trg_in=$raw_data_trg
  > src
  > trg
  :: SRC=@ TRG=@
  :: tokenize_protect=@
  :: tokenizer=@
  :: .submitter=$submitter .action_flags=$action_flags .resource_flags=$resource_flags {

  if [[ -z $tokenize_protect ]]; then
    tokenize_protect=${mosesdecoder}/scripts/tokenizer/basic-protected-patterns
  fi

  for lang in $SRC $TRG; do
    if [[ $lang == $SRC ]]; then
      get=$src_in
      put=$src
    else
      get=$trg_in
      put=$trg
    fi
    if [[ $tokenizer == "cjk" ]]; then
        cat $get \
          | ${mosesdecoder}/scripts/tokenizer/normalize-punctuation.perl -l $lang \
          | ${sockeye_scripts}/cjk/segment-cjk.py \
          > $put &

    elif [[ $tokenizer == "default" && ( $lang == "zh" || $lang == "ja" || $lang == "ko" ) ]]; then
        cat $get | ${sockeye_scripts}/cjk/segment-cjk.py \
          | ${mosesdecoder}/scripts/tokenizer/normalize-punctuation.perl -l $lang \
          | ${mosesdecoder}/scripts/tokenizer/tokenizer.perl -l $lang -no-escape -protected ${tokenize_protect} -q \
          > $put &

    elif [[ $tokenizer == "kytea" && $lang == "ja" ]]; then
      kytea=~/code/kytea
      cat $get | ${kytea}/src/bin/kytea -model ${kytea}/data/model.bin -out tok > $put &

    else
      cat $get \
        | ${mosesdecoder}/scripts/tokenizer/normalize-punctuation.perl -l $lang \
        | ${mosesdecoder}/scripts/tokenizer/remove-non-printing-char.perl \
        | ${mosesdecoder}/scripts/tokenizer/tokenizer.perl -l $lang -no-escape -protected ${tokenize_protect} -q \
        > $put &
    fi
  done

  wait
}
