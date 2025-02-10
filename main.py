"""
Algorithm server definition.
Documentation: https://github.com/Imaging-Server-Kit/cookiecutter-serverkit
"""
from typing import List, Type
from pathlib import Path
import numpy as np
from pydantic import BaseModel, Field, field_validator
import uvicorn
import skimage.io
import imaging_server_kit as serverkit

from transformers import BlipProcessor, BlipForConditionalGeneration

class Parameters(BaseModel):
    """Defines the algorithm parameters"""
    image: str = Field(
        ...,
        title="Image",
        description="Input image (2D, RGB).",
        json_schema_extra={"widget_type": "image"},
    )

    conditional_text: str = Field(
        default="an image of",
        title="Text",
        description="Conditional text (beginning of the caption).",
        json_schema_extra={"widget_type": "str"},
    )

    @field_validator("image", mode="after")
    def decode_image_array(cls, v) -> np.ndarray:
        image_array = serverkit.decode_contents(v)
        if image_array.ndim not in [2, 3]:
            raise ValueError("Array has the wrong dimensionality.")
        return image_array

class Server(serverkit.Server):
    def __init__(
        self,
        algorithm_name: str="blip-captioning",
        parameters_model: Type[BaseModel]=Parameters
    ):
        super().__init__(algorithm_name, parameters_model)

    def run_algorithm(
        self,
        image: np.ndarray,
        conditional_text: str,
        **kwargs
    ) -> List[tuple]:
        """Runs the algorithm."""
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        
        inputs = processor(image, conditional_text, return_tensors="pt")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)

        return [(caption, {}, "text")]

    def load_sample_images(self) -> List["np.ndarray"]:
        """Loads one or multiple sample images."""
        image_dir = Path(__file__).parent / "sample_images"
        images = [skimage.io.imread(image_path) for image_path in image_dir.glob("*")]
        return images

server = Server()
app = server.app

if __name__=='__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=8000)