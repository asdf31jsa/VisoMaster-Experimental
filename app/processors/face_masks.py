from typing import TYPE_CHECKING

import torch
import numpy as np
from torchvision import transforms
from torchvision.transforms import v2
import kornia.morphology as morph
from collections import defaultdict

from app.processors.external.clipseg import CLIPDensePredT
from app.processors.models_data import models_dir
if TYPE_CHECKING:
    from app.processors.models_processor import ModelsProcessor

class FaceMasks:
    def __init__(self, models_processor: 'ModelsProcessor'):
        self.models_processor = models_processor

    def apply_occlusion(self, img, amount):
        img = torch.div(img, 255)
        img = torch.unsqueeze(img, 0).contiguous()
        outpred = torch.ones((256,256), dtype=torch.float32, device=self.models_processor.device).contiguous()

        self.models_processor.run_occluder(img, outpred)

        outpred = torch.squeeze(outpred)
        outpred = (outpred > 0)
        outpred = torch.unsqueeze(outpred, 0).type(torch.float32)

        if amount >0:
            kernel = torch.ones((1,1,3,3), dtype=torch.float32, device=self.models_processor.device)

            for _ in range(int(amount)):
                outpred = torch.nn.functional.conv2d(outpred, kernel, padding=(1, 1))
                outpred = torch.clamp(outpred, 0, 1)

            outpred = torch.squeeze(outpred)

        if amount <0:
            outpred = torch.neg(outpred)
            outpred = torch.add(outpred, 1)
            kernel = torch.ones((1,1,3,3), dtype=torch.float32, device=self.models_processor.device)

            for _ in range(int(-amount)):
                outpred = torch.nn.functional.conv2d(outpred, kernel, padding=(1, 1))
                outpred = torch.clamp(outpred, 0, 1)

            outpred = torch.squeeze(outpred)
            outpred = torch.neg(outpred)
            outpred = torch.add(outpred, 1)

        outpred = torch.reshape(outpred, (1, 256, 256))
        return outpred

    def run_occluder(self, image, output):
        if not self.models_processor.models['Occluder']:
            self.models_processor.models['Occluder'] = self.models_processor.load_model('Occluder')

        io_binding = self.models_processor.models['Occluder'].io_binding()
        io_binding.bind_input(name='img', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,3,256,256), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,1,256,256), buffer_ptr=output.data_ptr())

        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()
        self.models_processor.models['Occluder'].run_with_iobinding(io_binding)

    def apply_dfl_xseg(self, img, amount, mouth, parameters):
        amount2 = -parameters["DFLXSeg2SizeSlider"]
        
        img = img.type(torch.float32)
        img = torch.div(img, 255)
        img = torch.unsqueeze(img, 0).contiguous()
        outpred = torch.ones((256,256), dtype=torch.float32, device=self.models_processor.device).contiguous()

        self.run_dfl_xseg(img, outpred)

        outpred = torch.clamp(outpred, min=0.0, max=1.0)
        outpred[outpred < 0.1] = 0
        # invert values to mask areas to keep
        outpred = 1.0 - outpred
        outpred = torch.unsqueeze(outpred, 0).type(torch.float32)

        if amount2 != amount:
            outpred2 = outpred.clone()

        if amount > 0:
            kernel = torch.ones((1,1,3,3), dtype=torch.float32, device=self.models_processor.device)

            for _ in range(int(amount)):
                outpred = torch.nn.functional.conv2d(outpred, kernel, padding=(1, 1))
                outpred = torch.clamp(outpred, 0, 1)

            #outpred = torch.squeeze(outpred)

        if amount < 0:
            outpred = torch.neg(outpred)
            outpred = torch.add(outpred, 1)
            kernel = torch.ones((1,1,3,3), dtype=torch.float32, device=self.models_processor.device)

            for _ in range(int(-amount)):
                outpred = torch.nn.functional.conv2d(outpred, kernel, padding=(1, 1))
                outpred = torch.clamp(outpred, 0, 1)

            #outpred = torch.squeeze(outpred)
            outpred = torch.neg(outpred)
            outpred = torch.add(outpred, 1)
        
        gauss = transforms.GaussianBlur(parameters['OccluderXSegBlurSlider']*2+1, (parameters['OccluderXSegBlurSlider']+1)*0.2)
        outpred = gauss(outpred)  
        if amount2 != amount:
            if amount2 > 0:
                kernel2 = torch.ones((1,1,3,3), dtype=torch.float32, device=self.models_processor.device)

                for _ in range(int(amount2)):
                    outpred2 = torch.nn.functional.conv2d(outpred2, kernel2, padding=(1, 1))
                    outpred2 = torch.clamp(outpred2, 0, 1)

                #outpred2 = torch.squeeze(outpred2)

            if amount2 < 0:
                outpred2 = torch.neg(outpred2)
                outpred2 = torch.add(outpred2, 1)
                kernel2 = torch.ones((1,1,3,3), dtype=torch.float32, device=self.models_processor.device)

                for _ in range(int(-amount2)):
                    outpred2 = torch.nn.functional.conv2d(outpred2, kernel2, padding=(1, 1))
                    outpred2 = torch.clamp(outpred2, 0, 1)

                #outpred2 = torch.squeeze(outpred2)
                outpred2 = torch.neg(outpred2)
                outpred2 = torch.add(outpred2, 1)
                
            gauss = transforms.GaussianBlur(parameters['XSeg2BlurSlider']*2+1, (parameters['XSeg2BlurSlider']+1)*0.2)
            outpred2 = gauss(outpred2) 
            
            #print("outpred, outpred2, mouth: ", outpred.shape, outpred2.shape, mouth.shape)
            outpred[mouth > 0.9] = outpred2[mouth > 0.9]

        outpred = torch.reshape(outpred, (1, 256, 256))
        return outpred

    def run_dfl_xseg(self, image, output):
        if not self.models_processor.models['XSeg']:
            self.models_processor.models['XSeg'] = self.models_processor.load_model('XSeg')

        io_binding = self.models_processor.models['XSeg'].io_binding()
        io_binding.bind_input(name='in_face:0', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=image.size(), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='out_mask:0', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,1,256,256), buffer_ptr=output.data_ptr())

        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()
        self.models_processor.models['XSeg'].run_with_iobinding(io_binding)
        
    def apply_face_parser(self, img, parameters, mode):
        FaceAmount = -parameters["BackgroundParserSlider"]
        FaceAmountTexture = -parameters["BackgroundParserTextureSlider"]

        img = torch.div(img, 255.0)
        img = v2.functional.normalize(img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        img = img.reshape(1, 3, 512, 512)

        outpred = torch.empty((1, 19, 512, 512), dtype=torch.float32, device=self.models_processor.device)
        self.run_faceparser(img, outpred)

        outpred = torch.argmax(outpred.squeeze(0), 0)

        def create_mask(attributes, iterations):
            mask = torch.isin(outpred, torch.tensor(attributes, device=outpred.device)).float()
            if iterations < 0:
                mask = 1 - mask
            mask = mask.unsqueeze(0).unsqueeze(0)
            for _ in range(abs(iterations)):
                mask = morph.dilation(mask, kernel=torch.ones((3, 3), device=mask.device))
            if iterations < 0:
                mask = 1 - mask
            return mask.squeeze(0)

        face_attributes = {
            1: parameters['FaceParserSlider'],
            2: parameters['LeftEyebrowParserSlider'],
            3: parameters['RightEyebrowParserSlider'],
            4: parameters['LeftEyeParserSlider'],
            5: parameters['RightEyeParserSlider'],
            6: parameters['EyeGlassesParserSlider'],
            10: parameters['NoseParserSlider'],
            11: parameters['MouthParserSlider'],
            12: parameters['UpperLipParserSlider'],
            13: parameters['LowerLipParserSlider'],
            14: parameters['NeckParserSlider'],
            17: parameters['HairParserSlider'],
        }

        face_attributes_texture = {
            2: parameters['EyebrowParserTextureSlider'],
            3: parameters['EyebrowParserTextureSlider'],
            4: parameters['EyeParserTextureSlider'],
            5: parameters['EyeParserTextureSlider'],
            10: parameters['NoseParserTextureSlider'],
            11: parameters['MouthParserTextureSlider'],
            12: parameters['MouthParserTextureSlider'],
            13: parameters['MouthParserTextureSlider'],
            14: parameters['NeckParserTextureSlider'],
        }

        mouth_attributes = {
            11: parameters['XsegMouthParserSlider'],
            12: parameters['XsegUpperLipParserSlider'],
            13: parameters['XsegLowerLipParserSlider']
        }

        bg_attributes = [0, 14, 15, 16, 17, 18]
        bg_attributes_texture = [0, 14, 15, 16, 17, 18]

        def group_and_combine(attr_dict):
            result = torch.zeros((1, 512, 512), dtype=torch.float32, device=outpred.device)
            grouped = defaultdict(list)
            for attr, dil in attr_dict.items():
                if dil != 0:
                    grouped[dil].append(attr)
            for dil, attrs in grouped.items():
                result += create_mask(attrs, dil)
            return torch.clamp(result, 0, 1)

        out_parse = torch.zeros((1, 512, 512), dtype=torch.float32, device=outpred.device)
        if parameters["FaceParserEnableToggle"]:
            out_parse = group_and_combine(face_attributes)
            if parameters['FaceBlurParserSlider'] > 0:
                k = parameters['FaceBlurParserSlider'] * 2 + 1
                sigma = (parameters['FaceBlurParserSlider'] + 1) * 0.2
                out_parse = transforms.GaussianBlur(k, sigma)(out_parse)

        combined_mouth_mask = torch.zeros((1, 512, 512), dtype=torch.float32, device=outpred.device)
        if parameters["XSegMouthEnableToggle"]:
            combined_mouth_mask = group_and_combine(mouth_attributes)

        bg_parse = torch.zeros((1, 512, 512), dtype=torch.float32, device=outpred.device)
        if parameters["FaceParserEnableToggle"]:
            if FaceAmount != 0:
                bg_parse = create_mask(bg_attributes, FaceAmount)
                if parameters['BackgroundBlurParserSlider'] > 0:
                    k = parameters['BackgroundBlurParserSlider'] * 2 + 1
                    sigma = (parameters['BackgroundBlurParserSlider'] + 1) * 0.2
                    bg_parse = transforms.GaussianBlur(k, sigma)(bg_parse)

        out_parse_texture = torch.zeros((1, 512, 512), dtype=torch.float32, device=outpred.device)
        bg_parse_texture = torch.zeros((1, 512, 512), dtype=torch.float32, device=outpred.device)
        use_texture_mask = any(v != 0 for v in face_attributes_texture.values())
        if (parameters["TransferTextureEnableToggle"] or parameters["DifferencingEnableToggle"]) and parameters["ExcludeMaskEnableToggle"] and use_texture_mask:
            out_parse_texture = group_and_combine(face_attributes_texture)

            if parameters['FaceParserBlurTextureSlider'] > 0:
                k = parameters['FaceParserBlurTextureSlider'] * 2 + 1
                sigma = (parameters['FaceParserBlurTextureSlider'] + 1) * 0.2
                out_parse_texture = transforms.GaussianBlur(k, sigma)(out_parse_texture)
        use_bg_texture_mask = FaceAmountTexture != 0
        if (parameters["TransferTextureEnableToggle"] or parameters["DifferencingEnableToggle"]) and parameters["ExcludeMaskEnableToggle"] and mode == "original" and use_bg_texture_mask:
            bg_parse_texture = create_mask(bg_attributes_texture, FaceAmountTexture)
            
            if parameters['FaceParserBlurBGTextureSlider'] > 0:
                k = parameters['FaceParserBlurBGTextureSlider'] * 2 + 1
                sigma = (parameters['FaceParserBlurBGTextureSlider'] + 1) * 0.2
                bg_parse_texture = transforms.GaussianBlur(k, sigma)(bg_parse_texture)

        out_parse = 1 - torch.clamp(out_parse + bg_parse, 0, 1)
        face_mask = torch.clamp(out_parse_texture, 0, 1)
        bg_mask = torch.clamp(bg_parse_texture, 0, 1)

        return out_parse, face_mask, bg_mask, combined_mouth_mask

    '''
    def apply_face_parser(self, img, parameters):
        # Allgemeine Parameter
        FaceAmount = parameters["BackgroundParserSlider"]
        FaceAmountTexture = parameters["BackgroundParserTextureSlider"]
        FaceParserTextureSlider = parameters["FaceParserTextureSlider"]

        # Normalize the image
        img = torch.div(img, 255)
        img = v2.functional.normalize(img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        img = torch.reshape(img, (1, 3, 512, 512))
        outpred = torch.empty((1, 19, 512, 512), dtype=torch.float32, device=self.models_processor.device).contiguous()

        self.run_faceparser(img, outpred)

        outpred = torch.squeeze(outpred)
        outpred = torch.argmax(outpred, 0)

        # Attribute-Slider-Zuordnung
        face_attributes = {
            1: parameters['FaceParserSlider'],  # Face
            2: parameters['LeftEyebrowParserSlider'],  # Left Eyebrow
            3: parameters['RightEyebrowParserSlider'],  # Right Eyebrow
            4: parameters['LeftEyeParserSlider'],  # Left Eye
            5: parameters['RightEyeParserSlider'],  # Right Eye
            6: parameters['EyeGlassesParserSlider'],  # EyeGlasses
            10: parameters['NoseParserSlider'],  # Nose
            11: parameters['MouthParserSlider'],  # Mouth
            12: parameters['UpperLipParserSlider'],  # Upper Lip
            13: parameters['LowerLipParserSlider'],  # Lower Lip
            14: parameters['NeckParserSlider'],  # Neck
            17: parameters['HairParserSlider'],  # Hair
        }

        texture_attributes = [2, 3, 4, 5, 10, 11, 12, 13, 14]  # Fixe Attribute für die Texturmaske

        # Pre-defined dilation kernel
        kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32, device=self.models_processor.device)

        # Maske für Face-Parsing (mit Gaussian Blur)
        face_parses = []
        for attribute, attribute_value in face_attributes.items():
            if attribute_value > 0:
                attribute_mask = torch.isin(outpred, torch.tensor([attribute], device=self.models_processor.device))
                attribute_mask = attribute_mask.float().unsqueeze(0).unsqueeze(0)  # (1, 1, 512, 512)
                for _ in range(int(attribute_value)):
                    attribute_mask = torch.nn.functional.conv2d(attribute_mask, kernel, padding=(1, 1))
                    attribute_mask = torch.clamp(attribute_mask, 0, 1)
                face_parses.append(attribute_mask)
            else:
                face_parses.append(torch.ones((1, 1, 512, 512), dtype=torch.float32, device=self.models_processor.device))

        combined_face_parse = torch.ones((1, 1, 512, 512), dtype=torch.float32, device=self.models_processor.device)
        for face_parse in face_parses:
            combined_face_parse = combined_face_parse * face_parse

        # Apply Gaussian blur to the combined face mask (einmal für alle)
        blur_kernel_size = parameters['FaceBlurParserSlider'] * 2 + 1
        if blur_kernel_size > 1:
            gauss = transforms.GaussianBlur(blur_kernel_size, (parameters['FaceBlurParserSlider'] + 1) * 0.2)
            combined_face_parse = gauss(combined_face_parse)

        # Hintergrundmaske für Face-Parsing
        bg_idxs = torch.tensor([0, 14, 15, 16, 17, 18], device=self.models_processor.device)
        bg_parse = 1 - torch.isin(outpred, bg_idxs).float().unsqueeze(0).unsqueeze(0)

        if FaceAmount > 0:
            for _ in range(int(FaceAmount)):
                bg_parse = torch.nn.functional.conv2d(bg_parse, kernel, padding=(1, 1))
                bg_parse = torch.clamp(bg_parse, 0, 1)
            if blur_kernel_size > 1:
                bg_parse = gauss(bg_parse)
        elif FaceAmount < 0:
            bg_parse = 1 - bg_parse
            for _ in range(int(-FaceAmount)):
                bg_parse = torch.nn.functional.conv2d(bg_parse, kernel, padding=(1, 1))
                bg_parse = torch.clamp(bg_parse, 0, 1)
            bg_parse = 1 - bg_parse
            if blur_kernel_size > 1:
                bg_parse = gauss(bg_parse)

        out_parse = torch.clamp(combined_face_parse * bg_parse, 0, 1).squeeze(0)

        # Maske für die Textur-Parsing (keine Gaussian Blur, einheitliche Dilation)
        texture_mask = torch.isin(outpred, torch.tensor(texture_attributes, device=self.models_processor.device))
        texture_mask = texture_mask.float().unsqueeze(0).unsqueeze(0)

        if FaceParserTextureSlider > 0:
            for _ in range(int(FaceParserTextureSlider)):
                texture_mask = torch.nn.functional.conv2d(texture_mask, kernel, padding=(1, 1))
                texture_mask = torch.clamp(texture_mask, 0, 1)

        # Hintergrundmaske für die Textur
        bg_parse_texture = 1 - torch.isin(outpred, bg_idxs).float().unsqueeze(0).unsqueeze(0)

        if FaceAmountTexture > 0:
            for _ in range(int(FaceAmountTexture)):
                bg_parse_texture = torch.nn.functional.conv2d(bg_parse_texture, kernel, padding=(1, 1))
                bg_parse_texture = torch.clamp(bg_parse_texture, 0, 1)

        out_parse_texture = torch.clamp(texture_mask * bg_parse_texture, 0, 1).squeeze(0)

        return out_parse, out_parse_texture
    '''
    # https://github.com/yakhyo/face-parsing
    def run_faceparser(self, image, output):
        if not self.models_processor.models['FaceParser']:
            self.models_processor.models['FaceParser'] = self.models_processor.load_model('FaceParser')

        image = image.contiguous()
        io_binding = self.models_processor.models['FaceParser'].io_binding()
        io_binding.bind_input(name='input', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,3,512,512), buffer_ptr=image.data_ptr())
        io_binding.bind_output(name='output', device_type=self.models_processor.device, device_id=0, element_type=np.float32, shape=(1,19,512,512), buffer_ptr=output.data_ptr())

        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()
        self.models_processor.models['FaceParser'].run_with_iobinding(io_binding)

    def run_CLIPs(self, img, CLIPText, CLIPAmount):
        # Ottieni il dispositivo su cui si trova l'immagine
        device = img.device

        # Controllo se la sessione CLIP è già stata inizializzata
        if not self.models_processor.clip_session:
            self.models_processor.clip_session = CLIPDensePredT(version='ViT-B/16', reduce_dim=64, complex_trans_conv=True)
            self.models_processor.clip_session.eval()
            self.models_processor.clip_session.load_state_dict(torch.load(f'{models_dir}/rd64-uni-refined.pth', weights_only=True), strict=False)
            self.models_processor.clip_session.to(device)  # Sposta il modello sul dispositivo dell'immagine

        # Crea un mask tensor direttamente sul dispositivo dell'immagine
        clip_mask = torch.ones((352, 352), device=device)

        # L'immagine è già un tensore, quindi la converto a float32 e la normalizzo nel range [0, 1]
        img = img.float() / 255.0  # Conversione in float32 e normalizzazione

        # Rimuovi la parte ToTensor(), dato che img è già un tensore.
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize((352, 352))
        ])

        # Applica la trasformazione all'immagine
        CLIPimg = transform(img).unsqueeze(0).contiguous().to(device)

        # Se ci sono prompt CLIPText, esegui la predizione
        if CLIPText != "":
            prompts = CLIPText.split(',')

            with torch.no_grad():
                # Esegui la predizione sulla sessione CLIP
                preds = self.models_processor.clip_session(CLIPimg.repeat(len(prompts), 1, 1, 1), prompts)[0]

            # Calcola la maschera CLIP usando la sigmoid e tieni tutto sul dispositivo
            clip_mask = 1 - torch.sigmoid(preds[0][0])
            for i in range(len(prompts) - 1):
                clip_mask *= 1 - torch.sigmoid(preds[i + 1][0])

            # Applica la soglia sulla maschera
            thresh = CLIPAmount / 100.0
            clip_mask = (clip_mask > thresh).float()

        return clip_mask.unsqueeze(0)  # Ritorna il tensore torch direttamente

    def soft_oval_mask(self, height, width, center, radius_x, radius_y, feather_radius=None):
        """
        Create a soft oval mask with feathering effect using integer operations.

        Args:
            height (int): Height of the mask.
            width (int): Width of the mask.
            center (tuple): Center of the oval (x, y).
            radius_x (int): Radius of the oval along the x-axis.
            radius_y (int): Radius of the oval along the y-axis.
            feather_radius (int): Radius for feathering effect.

        Returns:
            torch.Tensor: Soft oval mask tensor of shape (H, W).
        """
        if feather_radius is None:
            feather_radius = max(radius_x, radius_y) // 2  # Integer division

        # Calculating the normalized distance from the center
        y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')

        # Calculating the normalized distance from the center
        normalized_distance = torch.sqrt(((x - center[0]) / radius_x) ** 2 + ((y - center[1]) / radius_y) ** 2)

        # Creating the oval mask with a feathering effect
        mask = torch.clamp((1 - normalized_distance) * (radius_x / feather_radius), 0, 1)

        return mask

    def restore_mouth(self, img_orig, img_swap, kpss_orig, blend_alpha=0.5, feather_radius=10, size_factor=0.5, radius_factor_x=1.0, radius_factor_y=1.0, x_offset=0, y_offset=0):
        """
        Extract mouth from img_orig using the provided keypoints and place it in img_swap.

        Args:
            img_orig (torch.Tensor): The original image tensor of shape (C, H, W) from which mouth is extracted.
            img_swap (torch.Tensor): The target image tensor of shape (C, H, W) where mouth is placed.
            kpss_orig (list): List of keypoints arrays for detected faces. Each keypoints array contains coordinates for 5 keypoints.
            radius_factor_x (float): Factor to scale the horizontal radius. 1.0 means circular, >1.0 means wider oval, <1.0 means narrower.
            radius_factor_y (float): Factor to scale the vertical radius. 1.0 means circular, >1.0 means taller oval, <1.0 means shorter.
            x_offset (int): Horizontal offset for shifting the mouth left (negative value) or right (positive value).
            y_offset (int): Vertical offset for shifting the mouth up (negative value) or down (positive value).

        Returns:
            torch.Tensor: The resulting image tensor with mouth from img_orig placed on img_swap.
        """
        left_mouth = np.array([int(val) for val in kpss_orig[3]])
        right_mouth = np.array([int(val) for val in kpss_orig[4]])

        mouth_center = (left_mouth + right_mouth) // 2
        mouth_base_radius = int(np.linalg.norm(left_mouth - right_mouth) * size_factor)

        # Calculate the scaled radii
        radius_x = int(mouth_base_radius * radius_factor_x)
        radius_y = int(mouth_base_radius * radius_factor_y)

        # Apply the x/y_offset to the mouth center
        mouth_center[0] += x_offset
        mouth_center[1] += y_offset

        # Calculate bounding box for mouth region
        ymin = max(0, mouth_center[1] - radius_y)
        ymax = min(img_orig.size(1), mouth_center[1] + radius_y)
        xmin = max(0, mouth_center[0] - radius_x)
        xmax = min(img_orig.size(2), mouth_center[0] + radius_x)

        mouth_region_orig = img_orig[:, ymin:ymax, xmin:xmax]
        mouth_mask = self.soft_oval_mask(ymax - ymin, xmax - xmin,
                                            (radius_x, radius_y),
                                            radius_x, radius_y,
                                            feather_radius).to(img_orig.device)

        target_ymin = ymin
        target_ymax = ymin + mouth_region_orig.size(1)
        target_xmin = xmin
        target_xmax = xmin + mouth_region_orig.size(2)

        img_swap_mouth = img_swap[:, target_ymin:target_ymax, target_xmin:target_xmax]
        blended_mouth = blend_alpha * img_swap_mouth + (1 - blend_alpha) * mouth_region_orig

        img_swap[:, target_ymin:target_ymax, target_xmin:target_xmax] = mouth_mask * blended_mouth + (1 - mouth_mask) * img_swap_mouth
        return img_swap

    def restore_eyes(self, img_orig, img_swap, kpss_orig, blend_alpha=0.5, feather_radius=10, size_factor=3.5, radius_factor_x=1.0, radius_factor_y=1.0, x_offset=0, y_offset=0, eye_spacing_offset=0):
        """
        Extract eyes from img_orig using the provided keypoints and place them in img_swap.

        Args:
            img_orig (torch.Tensor): The original image tensor of shape (C, H, W) from which eyes are extracted.
            img_swap (torch.Tensor): The target image tensor of shape (C, H, W) where eyes are placed.
            kpss_orig (list): List of keypoints arrays for detected faces. Each keypoints array contains coordinates for 5 keypoints.
            radius_factor_x (float): Factor to scale the horizontal radius. 1.0 means circular, >1.0 means wider oval, <1.0 means narrower.
            radius_factor_y (float): Factor to scale the vertical radius. 1.0 means circular, >1.0 means taller oval, <1.0 means shorter.
            x_offset (int): Horizontal offset for shifting the eyes left (negative value) or right (positive value).
            y_offset (int): Vertical offset for shifting the eyes up (negative value) or down (positive value).
            eye_spacing_offset (int): Horizontal offset to move eyes closer together (negative value) or farther apart (positive value).

        Returns:
            torch.Tensor: The resulting image tensor with eyes from img_orig placed on img_swap.
        """
        # Extract original keypoints for left and right eye
        left_eye = np.array([int(val) for val in kpss_orig[0]])
        right_eye = np.array([int(val) for val in kpss_orig[1]])

        # Apply horizontal offset (x-axis)
        left_eye[0] += x_offset
        right_eye[0] += x_offset

        # Apply vertical offset (y-axis)
        left_eye[1] += y_offset
        right_eye[1] += y_offset

        # Calculate eye distance and radii
        eye_distance = np.linalg.norm(left_eye - right_eye)
        base_eye_radius = int(eye_distance / size_factor)

        # Calculate the scaled radii
        radius_x = int(base_eye_radius * radius_factor_x)
        radius_y = int(base_eye_radius * radius_factor_y)

        # Adjust for eye spacing (horizontal movement)
        left_eye[0] += eye_spacing_offset
        right_eye[0] -= eye_spacing_offset

        def extract_and_blend_eye(eye_center, radius_x, radius_y, img_orig, img_swap, blend_alpha, feather_radius):
            ymin = max(0, eye_center[1] - radius_y)
            ymax = min(img_orig.size(1), eye_center[1] + radius_y)
            xmin = max(0, eye_center[0] - radius_x)
            xmax = min(img_orig.size(2), eye_center[0] + radius_x)

            eye_region_orig = img_orig[:, ymin:ymax, xmin:xmax]
            eye_mask = self.soft_oval_mask(ymax - ymin, xmax - xmin,
                                            (radius_x, radius_y),
                                            radius_x, radius_y,
                                            feather_radius).to(img_orig.device)

            target_ymin = ymin
            target_ymax = ymin + eye_region_orig.size(1)
            target_xmin = xmin
            target_xmax = xmin + eye_region_orig.size(2)

            img_swap_eye = img_swap[:, target_ymin:target_ymax, target_xmin:target_xmax]
            blended_eye = blend_alpha * img_swap_eye + (1 - blend_alpha) * eye_region_orig

            img_swap[:, target_ymin:target_ymax, target_xmin:target_xmax] = eye_mask * blended_eye + (1 - eye_mask) * img_swap_eye

        # Process both eyes with updated positions
        extract_and_blend_eye(left_eye, radius_x, radius_y, img_orig, img_swap, blend_alpha, feather_radius)
        extract_and_blend_eye(right_eye, radius_x, radius_y, img_orig, img_swap, blend_alpha, feather_radius)

        return img_swap

    def apply_fake_diff(self, swapped_face, original_face, lower_thresh, lower_value, upper_thresh, upper_value, middle_value):
        # Kein permute nötig → [3, H, W]
        diff = torch.abs(swapped_face - original_face)

        # Quantile (auf allen Kanälen)
        sample = diff.reshape(-1)
        sample = sample[torch.randint(0, sample.numel(), (50_000,), device=diff.device)]
        diff_max = torch.quantile(sample, 0.99)
        diff = torch.clamp(diff, max=diff_max)

        diff_min = diff.min()
        diff_max = diff.max()
        diff_norm = (diff - diff_min) / (diff_max - diff_min)

        diff_mean = diff_norm.mean(dim=0)  # [H, W]

        # Direkt mit torch.where statt vielen Masken
        scale = diff_mean / lower_thresh
        result = torch.where(
            diff_mean < lower_thresh,
            lower_value + scale * (middle_value - lower_value),
            torch.empty_like(diff_mean)
        )

        middle_scale = (diff_mean - lower_thresh) / (upper_thresh - lower_thresh)
        result = torch.where(
            (diff_mean >= lower_thresh) & (diff_mean <= upper_thresh),
            middle_value + middle_scale * (upper_value - middle_value),
            result
        )

        above_scale = (diff_mean - upper_thresh) / (1 - upper_thresh)
        result = torch.where(
            diff_mean > upper_thresh,
            upper_value + above_scale * (1 - upper_value),
            result
        )

        return result.unsqueeze(0)  # (1, H, W)


