task fast_align : fast_align
    < src_in=$out@download_or_link[DataSection:train,side:src]
    < trg_in=$out@download_or_link[DataSection:train,side:trg]
    > forward_log="log.forward"
    > reverse_log="log.reverse"
    > forward_lexicon="lexicon.forward"
    > reverse_lexicon="lexicon.reverse"
    :: .submitter=$submitter
    :: .resource_flags=$resource_flags_16g
    :: .action_flags=@ {
  zpaste $src_in $trg_in | perl -pe 's/\t/ ||| /g' > corpus

  ${fast_align}/build/src/fast_align -i corpus -v -d -o -p $forward_lexicon 2> $forward_log | gzip -9n > alignments.forward.gz
  ${fast_align}/build/src/fast_align -i corpus -v -d -o -r -p $reverse_lexicon 2> $reverse_log | gzip -9n > alignments.reverse.gz
}
