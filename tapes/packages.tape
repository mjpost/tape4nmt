package sockeye :: .versioner=git .repo="https://github.com/awslabs/sockeye" .ref=HEAD { }
package subword_nmt :: .versioner=pip .package="subword-nmt" .tag="0.3.6" { }
package sockeye_scripts :: .versioner=git .repo="/home/hltcoe/mpost/code/sockeye-scripts" .ref=HEAD { }
package fast_align :: .versioner=git .repo="https://github.com/clab/fast_align" .ref=HEAD { 
  mkdir build
  cd build
  cmake ..
  make -j $(nproc)

  # the force_align.py script looks in the wrong spot
  cp fast_align atools ../src
}

package mosesdecoder :: .versioner=git .repo="https://github.com/moses-smt/mosesdecoder" .ref=HEAD { 
  # This is needed to prevent the Moses tokenizer from splitting out masks
  echo '__[A-Za-z0-9]+(_\d+)?__' >> scripts/tokenizer/basic-protected-patterns
}

package sentencepiece :: .versioner=git .repo="https://github.com/google/sentencepiece" .ref=HEAD {
  mkdir build
  cd build
  cmake ..
  make -j $(nproc)
}

package tools :: .versioner=git .repo="https://github.com/shuoyangd/tape4nmt-tools" .ref=HEAD {
  git submodule update --init --recursive
  pip install -r requirements.txt
}

# common tools (unpaste, debpe)
package bin :: .versioner=git .repo="https://github.com/mjpost/bin" .ref=HEAD { }

package fairseq :: .versioner=git .repo="https://github.com/mjpost/fairseq" .ref="stable" {
  python setup.py build develop
}

