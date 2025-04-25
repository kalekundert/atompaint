from lightning import Callback

# - pseudocode:
#   - iterate through validation set
#   - remove sidechains from image
#   - make mask that keeps only backbone atoms
#   - inpaint; very expensive
#   - make amino acid predictions
#   - calculate sequence recovery
#
# - This would be just as expensive as running the generation metrics, so it 
#   wouldn't be practical to evaluate the whole validation set like this.
#   I also probably can't afford to run both.
#
#   - Wait, I think this is more expensive.  Because inpainting has backwards 
#     steps.  
#
# - Perhaps I could make a callback to do this; although that callback would 
#   need access to the dataset, which feels wrong.  That said, this is the 
#   value I want to optimize for...
#
#   - Actually, `on_validation_batch_end()` gives access to the batches used 
#     in validation.  So I could implement that, and just stop after I've 
#     seen enough.
#
#  - I'll need to write this code at some point, anyways.  Might as well 
#    write it now, and use it to validate models.
#
#  - First, though, let me just get the training script running.  Then I can 
#    commit the end-to-end dataset code, and make a new commit for the 
#    callback.
#

class SequenceRecovery(Callback):

    pass

    
