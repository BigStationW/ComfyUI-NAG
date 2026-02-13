import torch
import comfy
from comfy_extras.nodes_custom_sampler import Noise_EmptyNoise, Noise_RandomNoise
import latent_preview

from .samplers import NAGCFGGuider as samplers_NAGCFGGuider
from .sample import sample_with_nag, sample_custom_with_nag


def common_ksampler_with_nag(model, seed, steps, cfg, nag_scale, nag_tau, nag_alpha, nag_sigma_end, sampler_name,
                             scheduler, positive, negative, nag_negative, latent, denoise=1.0, disable_noise=False,
                             start_step=None, last_step=None, force_full_denoise=False, **kwargs):
    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)

    if disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
    else:
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    callback = latent_preview.prepare_callback(model, steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    samples = sample_with_nag(
        model, noise, steps, cfg, nag_scale, nag_tau, nag_alpha, 14.7, nag_sigma_end, sampler_name, scheduler, positive,
        negative, nag_negative, latent_image,
        denoise=denoise, disable_noise=disable_noise, start_step=start_step, last_step=last_step,
        force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar,
        seed=seed, **kwargs,
    )
    out = latent.copy()
    out["samples"] = samples
    return (out,)


class NAGGuider:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
            {
                "model": ("MODEL",),
                "conditioning": ("CONDITIONING",),
                "nag_negative": ("CONDITIONING",),
                "nag_scale": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "nag_tau": ("FLOAT", {"default": 2.5, "min": 1.0, "max": 10.0, "step": 0.1, "round": 0.01}),
                "nag_alpha": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.01}),
                "nag_sigma_end": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 20.0, "step": 0.01, "round": 0.01}),
                "latent_image": ("LATENT",),
            }
        }

    RETURN_TYPES = ("GUIDER",)

    FUNCTION = "get_guider"
    CATEGORY = "sampling/custom_sampling/guiders"

    def get_guider(
            self,
            model,
            conditioning,
            nag_negative,
            nag_scale,
            nag_tau,
            nag_alpha,
            nag_sigma_end,
            latent_image,
    ):
        batch_size = latent_image["samples"].shape[0]
        guider = samplers_NAGCFGGuider(model)
        guider.set_conds(conditioning)
        guider.set_batch_size(batch_size)
        guider.set_nag(nag_negative, nag_scale, nag_tau, nag_alpha, 14.7, nag_sigma_end)
        return (guider,)


class NAGCFGGuider:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
            {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "nag_negative": ("CONDITIONING",),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "nag_scale": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "nag_tau": ("FLOAT", {"default": 2.5, "min": 1.0, "max": 10.0, "step": 0.1, "round": 0.01}),
                "nag_alpha": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.01}),
                "nag_sigma_end": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 20.0, "step": 0.01, "round": 0.01}),
                "latent_image": ("LATENT",),
            }
        }

    RETURN_TYPES = ("GUIDER",)

    FUNCTION = "get_guider"
    CATEGORY = "sampling/custom_sampling/guiders"

    def get_guider(
            self,
            model,
            positive,
            negative,
            nag_negative,
            cfg,
            nag_scale,
            nag_tau,
            nag_alpha,
            nag_sigma_end,
            latent_image,
    ):
        batch_size = latent_image["samples"].shape[0]
        guider = samplers_NAGCFGGuider(model)
        guider.set_conds(positive, negative)
        guider.set_cfg(cfg)
        guider.set_batch_size(batch_size)
        guider.set_nag(nag_negative, nag_scale, nag_tau, nag_alpha, 14.7, nag_sigma_end)
        return (guider,)


class KSamplerWithNAG:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model used for denoising the input latent."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True,
                                 "tooltip": "The random seed used for creating the noise."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000,
                                  "tooltip": "The number of steps used in the denoising process."}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01,
                                  "tooltip": "The Classifier-Free Guidance scale balances creativity and adherence to the prompt. Higher values result in images more closely matching the prompt however too high values will negatively impact quality."}),
                "nag_scale": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "nag_tau": ("FLOAT", {"default": 2.5, "min": 1.0, "max": 10.0, "step": 0.1, "round": 0.01}),
                "nag_alpha": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.01}),
                "nag_sigma_end": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 20.0, "step": 0.01, "round": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {
                    "tooltip": "The algorithm used when sampling, this can affect the quality, speed, and style of the generated output."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,
                              {"tooltip": "The scheduler controls how noise is gradually removed to form the image."}),
                "positive": ("CONDITIONING", {
                    "tooltip": "The conditioning describing the attributes you want to include in the image."}),
                "negative": ("CONDITIONING", {
                    "tooltip": "The conditioning describing the attributes you want to exclude from the image."}),
                "nag_negative": ("CONDITIONING", {
                    "tooltip": "The conditioning describing the attributes you want to exclude from the image for NAG."}),
                "latent_image": ("LATENT", {"tooltip": "The latent image to denoise."}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                      "tooltip": "The amount of denoising applied, lower values will maintain the structure of the initial image allowing for image to image sampling."}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    OUTPUT_TOOLTIPS = ("The denoised latent.",)
    FUNCTION = "sample"

    CATEGORY = "sampling"
    DESCRIPTION = "Uses the provided model, positive and negative conditioning to denoise the latent image."

    def sample(self, model, seed, steps, cfg, nag_scale, nag_tau, nag_alpha, nag_sigma_end, sampler_name, scheduler,
               positive, negative, nag_negative, latent_image, denoise=1.0):
        return common_ksampler_with_nag(model, seed, steps, cfg, nag_scale, nag_tau, nag_alpha, nag_sigma_end,
                                        sampler_name, scheduler, positive, negative, nag_negative, latent_image,
                                        denoise=denoise)


class KSamplerAdvancedWithNAG:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "add_noise": (["enable", "disable"],),
                "noise_seed": (
                    "INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "nag_scale": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "nag_tau": ("FLOAT", {"default": 2.5, "min": 1.0, "max": 10.0, "step": 0.1, "round": 0.01}),
                "nag_alpha": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.01}),
                "nag_sigma_end": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 20.0, "step": 0.01, "round": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "nag_negative": ("CONDITIONING", {
                    "tooltip": "The conditioning describing the attributes you want to exclude from the image for NAG."}),
                "latent_image": ("LATENT",),
                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                "return_with_leftover_noise": (["disable", "enable"],),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    CATEGORY = "sampling"

    def sample(
            self, model, add_noise, noise_seed, steps, cfg, nag_scale, nag_tau, nag_alpha, nag_sigma_end,
            sampler_name, scheduler, positive, negative, nag_negative,
            latent_image, start_at_step, end_at_step, return_with_leftover_noise, denoise=1.0,
    ):
        force_full_denoise = True
        if return_with_leftover_noise == "enable":
            force_full_denoise = False
        disable_noise = False
        if add_noise == "disable":
            disable_noise = True
        return common_ksampler_with_nag(
            model, noise_seed, steps, cfg, nag_scale, nag_tau, nag_alpha, nag_sigma_end,
            sampler_name, scheduler, positive, negative, nag_negative,
            latent_image,
            denoise=denoise, disable_noise=disable_noise, start_step=start_at_step,
            last_step=end_at_step, force_full_denoise=force_full_denoise,
        )


class SamplerCustomWithNAG:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL",),
            "add_noise": ("BOOLEAN", {"default": True}),
            "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
            "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
            "nag_scale": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
            "nag_tau": ("FLOAT", {"default": 2.5, "min": 1.0, "max": 10.0, "step": 0.1, "round": 0.01}),
            "nag_alpha": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.01}),
            "nag_sigma_end": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 20.0, "step": 0.01, "round": 0.01}),
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "nag_negative": ("CONDITIONING", {
                "tooltip": "The conditioning describing the attributes you want to exclude from the image for NAG."}),
            "sampler": ("SAMPLER",),
            "sigmas": ("SIGMAS",),
            "latent_image": ("LATENT",),
        }}

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("output", "denoised_output")

    FUNCTION = "sample"

    CATEGORY = "sampling/custom_sampling"

    def sample(
            self,
            model, add_noise, noise_seed, cfg, nag_scale, nag_tau, nag_alpha, nag_sigma_end,
            positive, negative, nag_negative,
            sampler, sigmas, latent_image,
    ):
        latent = latent_image
        latent_image = latent["samples"]
        latent = latent.copy()
        latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)
        latent["samples"] = latent_image

        if not add_noise:
            noise = Noise_EmptyNoise().generate_noise(latent)
        else:
            noise = Noise_RandomNoise(noise_seed).generate_noise(latent)

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]

        x0_output = {}
        callback = latent_preview.prepare_callback(model, sigmas.shape[-1] - 1, x0_output)

        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        samples = sample_custom_with_nag(
            model, noise, cfg, nag_scale, nag_tau, nag_alpha, 14.7, nag_sigma_end,
            sampler, sigmas, positive, negative, nag_negative,
            latent_image,
            noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise_seed,
        )

        out = latent.copy()
        out["samples"] = samples
        if "x0" in x0_output:
            out_denoised = latent.copy()
            out_denoised["samples"] = model.model.process_latent_out(x0_output["x0"].cpu())
        else:
            out_denoised = out
        return (out, out_denoised)
    
class NAGCFGGuiderAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
            {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "nag_negative": ("CONDITIONING",),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "nag_scale": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "nag_tau": ("FLOAT", {"default": 2.5, "min": 1.0, "max": 10.0, "step": 0.1, "round": 0.01}),
                "nag_alpha": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.01}),
                "nag_sigma_start": ("FLOAT", {"default": 14.7, "min": 0.0, "max": 20.0, "step": 0.01, "round": 0.01}),
                "nag_sigma_end": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 20.0, "step": 0.01, "round": 0.01}),
                "latent_image": ("LATENT",),
                "display_logs": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("GUIDER",)

    FUNCTION = "get_guider"
    CATEGORY = "sampling/custom_sampling/guiders"

    def get_guider(
            self,
            model,
            positive,
            negative,
            nag_negative,
            cfg,
            nag_scale,
            nag_tau,
            nag_alpha,
            nag_sigma_start,
            nag_sigma_end,
            latent_image,
            display_logs,
    ):
        batch_size = latent_image["samples"].shape[0]
        guider = samplers_NAGCFGGuider(model)
        guider.set_conds(positive, negative)
        guider.set_cfg(cfg)
        guider.set_batch_size(batch_size)
        guider.set_nag(nag_negative, nag_scale, nag_tau, nag_alpha, nag_sigma_start, nag_sigma_end)
        guider.set_display_logs(display_logs)
        return (guider,)


NODE_CLASS_MAPPINGS = {
    "NAGGuider": NAGGuider,
    "NAGCFGGuider": NAGCFGGuider,
    "NAGCFGGuiderAdvanced": NAGCFGGuiderAdvanced,
    "KSamplerWithNAG": KSamplerWithNAG,
    "KSamplerWithNAG (Advanced)": KSamplerAdvancedWithNAG,
    "SamplerCustomWithNAG": SamplerCustomWithNAG,
}
