"""
Design macromolecular interactions by in-painting full-atom models.
"""
__version__ = '0.1.0'

# Avoid importing any subpackages here.  This speeds up load time, since it 
# guarantees that no expensive packages (e.g. torch, lightning, escnn, pandas, 
# matplotlib, etc.) are imported unless they are needed.  This does mean that 
# users will have to be more specific about what they want to import, though.
