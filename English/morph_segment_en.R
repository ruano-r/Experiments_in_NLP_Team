#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(morphemepiece))

args <- commandArgs(trailingOnly = TRUE)
input_file  <- args[1]
output_file <- args[2]

lines <- readLines(input_file, encoding = "UTF-8", warn = FALSE)
words <- unique(unlist(strsplit(tolower(lines), "[^a-z]+")))
words <- words[nzchar(words)]
cat("Unique words:", length(words), "\n")

segment <- function(w) {
  p <- names(unlist(morphemepiece_tokenize(w)))
  if (length(p) == 0 || any(p == "[UNK]")) return(NA_character_)
  p <- gsub("^##|##$", "", p)       # strip ## from either side of a piece
  paste(p, collapse = "@@")
}

segs <- vapply(words, segment, character(1), USE.NAMES = FALSE)
segs <- segs[!is.na(segs)]
writeLines(segs, output_file, useBytes = TRUE)
cat("Wrote", length(segs), "segmentations\n")