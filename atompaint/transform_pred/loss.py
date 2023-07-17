import torch

class CoordFrameMseLoss(torch.nn.Module):
    """
    Calculate the (squared) distance between two coordinate frames.

    Specifically, this calculation is done by creating "probes" on the x-, y-, 
    and z-axes that are then transformed into both coordinate frames.  The mean 
    square distance between the two transformed versions of each probe is the 
    value calculated.

    The relative importance of rotation vs. translation can be controlled via 
    the *radius* parameter.  This determines how far from the origin each probe 
    is located, in units of Angstroms.  The further away they are, the greater 
    the lever-arm effect from rotations will be, and the more to overall score 
    will be influenced by the rotational component of the transformation.  
    Typically, the radius is set based on the size of the region visible to the 
    model (e.g. if the model can see a 10Å box, a radius of 5Å might be 
    appropriate, or at least a good starting point).
    """

    def __init__(self, radius_A):
        super().__init__()
        self.radius_A = radius_A

    def forward(self, predicted_frame, expected_frame):
        """
        Arguments:
            predicted_frame:
                The coordinate frames predicted by the context encoder, as 
                tensors of dimension (B, 4, 4). 

                B: minibatch size
                4,4: 3D roto-translation matrix

            expected_frame:
                The true coordinate frames, as tensors of dimension (B, 4, 4).
        """
        xyz = torch.cat([
                torch.eye(3) * self.radius_A,
                torch.ones((1,3)),
        ])
        xyz_pred = predicted_frame @ xyz
        xyz_expect = expected_frame @ xyz

        return torch.mean(torch.sum((xyz_pred - xyz_expect)**2, axis=1))
