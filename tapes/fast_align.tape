task fast_align : fast_align
  < src_in=$out@download_or_link[DataSection:train,side:src]
  < trg_in=$out@download_or_link[DataSection:train,side:trg]
  > alignment
  > lexicon
  :: .submitter=$submitter {

  zpaste $src_in $trg_in | perl -pe 's/\t/ ||| /g' > corpus

  ${fast_align}/build/src/fast_align -i corpus -v -d -o -p lexicon.forward 2> log.forward | gzip -9n > alignments.forward.gz
  ${fast_align}/build/src/fast_align -i corpus -v -d -o -p lexicon.reverse 2> log.reverse | gzip -9n > alignments.reverse.gz
}
