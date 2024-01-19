import { HfInference, HfInferenceEndpoint } from "@huggingface/inference";
import fs from "fs";

const inference = new HfInference();

async function main() {
  // You can also omit "model" to use the recommended model for the task
  const translation = await inference.translation({
    model: "t5-base",
    inputs: "My name is Wolfgang and I live in Amsterdam",
  });

  const image = await inference.textToImage({
    model: "stabilityai/stable-diffusion-2-1",
    inputs:
      "award winning high resolution photo of fighter jet cockpit mid flight",
    parameters: {
      negative_prompt: "blurry",
    },
  });

  console.log(translation);

  const buffer = Buffer.from(await image.arrayBuffer());
  fs.writeFileSync("image.jpg", buffer);

  const text = await inference.imageToText({
    data: await image.arrayBuffer(),
    model: "Salesforce/blip-image-captioning-large",
  });

  console.log(text);
}

main();
