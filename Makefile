ALIGNED_TXT = $(PWD)/src/labeling/class_aligned.txt
NOT_ALIGNED_TXT = $(PWD)/src/labeling/class_not_aligned.txt

TRAIN_DATA_ALIGNED_FOLDER = $(PWD)/data/train/aligned
TRAIN_DATA_NOT_ALIGNED_FOLDER = $(PWD)/data/train/not_aligned

symlinks:
	$(shell ./symlinks.sh $(ALIGNED_TXT) $(TRAIN_DATA_ALIGNED_FOLDER))
	$(shell ./symlinks.sh $(NOT_ALIGNED_TXT) $(TRAIN_DATA_NOT_ALIGNED_FOLDER))

clean:
	rm -f $(TRAIN_DATA_ALIGNED_FOLDER)/*
	rm -f $(TRAIN_DATA_NOT_ALIGNED_FOLDER)/*

run:


