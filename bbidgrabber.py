import copy
import math
import os
import random
import sys
import traceback
import shlex
import os, urllib.request, re, threading, posixpath, urllib.parse, argparse, socket, time, hashlib, pickle, signal, imghdr, io

import modules.scripts as scripts
import gradio as gr

from modules import sd_samplers
from modules.processing import Processed, process_images
from PIL import Image
from modules.shared import opts, cmd_opts, state


def process_string_tag(tag):
    return tag


def process_int_tag(tag):
    return int(tag)


def process_float_tag(tag):
    return float(tag)


def process_boolean_tag(tag):
    return True if (tag == "true") else False


prompt_tags = {
    "sd_model": None,
    "outpath_samples": process_string_tag,
    "outpath_grids": process_string_tag,
    "prompt_for_display": process_string_tag,
    "prompt": process_string_tag,
    "negative_prompt": process_string_tag,
    "styles": process_string_tag,
    "seed": process_int_tag,
    "subseed_strength": process_float_tag,
    "subseed": process_int_tag,
    "seed_resize_from_h": process_int_tag,
    "seed_resize_from_w": process_int_tag,
    "sampler_index": process_int_tag,
    "sampler_name": process_string_tag,
    "batch_size": process_int_tag,
    "n_iter": process_int_tag,
    "steps": process_int_tag,
    "cfg_scale": process_float_tag,
    "width": process_int_tag,
    "height": process_int_tag,
    "restore_faces": process_boolean_tag,
    "tiling": process_boolean_tag,
    "do_not_save_samples": process_boolean_tag,
    "do_not_save_grid": process_boolean_tag
}


def cmdargs(line):
    args = shlex.split(line)
    pos = 0
    res = {}

    while pos < len(args):
        arg = args[pos]

        assert arg.startswith("--"), f'must start with "--": {arg}'
        assert pos+1 < len(args), f'missing argument for command line option {arg}'

        tag = arg[2:]

        if tag == "prompt" or tag == "negative_prompt":
            pos += 1
            prompt = args[pos]
            pos += 1
            while pos < len(args) and not args[pos].startswith("--"):
                prompt += " "
                prompt += args[pos]
                pos += 1
            res[tag] = prompt
            continue


        func = prompt_tags.get(tag, None)
        assert func, f'unknown commandline option: {arg}'

        val = args[pos+1]
        if tag == "sampler_name":
            val = sd_samplers.samplers_map.get(val.lower(), None)

        res[tag] = func(val)

        pos += 2

    return res


def load_prompt_file(file):
    if file is None:
        return None, gr.update(), gr.update(lines=7)
    else:
        lines = [x.strip() for x in file.decode('utf8', errors='ignore').split("\n")]
        return None, "\n".join(lines), gr.update(lines=7)


class Script(scripts.Script):
    def title(self):
        return "Bing downloader"

    def show(self, is_img2img):
        return is_img2img
        
    def ui(self, is_img2img):
        genlabel = gr.HTML("<br> Will not work if there is no image inserted above.<br>Place a dummy image into the image picker before executing.<br><br>")
        bing_txt = gr.Textbox(label="Search Term", lines=1, elem_id=self.elem_id("bing_txt"))
        iterations_txt = gr.Slider(label="Images to produce", minimum=1, maximum=100, step=1, value=3, elem_id="iterations_txt")
        aspect_chk = gr.Checkbox(label="Set image output size to width x width * aspectratio", value=True, elem_id="aspect_chk")
        
        return [bing_txt, iterations_txt, aspect_chk]

    def run(self, p, bing_txt: str, iterations_txt: str, aspect_chk:bool):

        iterations = int(iterations_txt)

        bing_txt = bing_txt.lower().strip()
        texthash = str(hash(bing_txt))

        urlopenheader={ 'User-Agent' : 'Mozilla/5.0 (X11; Fedora; Linux x86_64; rv:60.0) Gecko/20100101 Firefox/60.0'}
        foundimage=None
        
        if not os.path.exists("./scripts/searchcache"):
            os.makedirs("./scripts/searchcache")
        
        if (os.path.isfile("./scripts/searchcache/"+texthash+".searchcache")):
            print ("ok")
            text_file = open("./scripts/searchcache/"+texthash+".searchcache", "r")
            totallinks = text_file.readlines()
            text_file.close()
            
        else:
            current = 0
            last = ''
            totallinks = []
            
            done = False
            
            while done == False:
                time.sleep(0.5)
                request_url='https://www.bing.com/images/async?q=' + urllib.parse.quote_plus(bing_txt) + '&first=' + str(current) + '&count=35&adlt=off'
                request=urllib.request.Request(request_url,None,headers=urlopenheader)
                response=urllib.request.urlopen(request)
                html = response.read().decode('utf8')
                links = re.findall('murl&quot;:&quot;(.*?)&quot;',html)
                
                if links[-1] == last or current > 20:
                    done = True
                    print("loop finished " + str(current))
                    break

                current += 1
                last = links[-1]
                totallinks.extend(links)

            text_file = open("./scripts/searchcache/"+texthash+".searchcache", "w")
            
            for l in totallinks:
                text_file.write(l + "\n")

            text_file.close()          

        p.do_not_save_grid = True

        job_count = 0
        jobs = []
        
        for i in range(iterations):
            args = {"prompt": p.prompt}        
            jobs.append(args)
            job_count += 1           
            
        state.job_count = job_count

        images = []
        all_prompts = []
        infotexts = []
        for n, args in enumerate(jobs):
            state.job = f"{state.job_no + 1} out of {state.job_count}"

            for tries in range(10):                           
            
                url = totallinks[random.randint(0, len(totallinks) - 1)]
                print ("downloading " + url)
                
                try:
                    request=urllib.request.Request(url,None,urlopenheader)
                    image_data=urllib.request.urlopen(request).read()
                except:
                    continue
                
                
                if not imghdr.what(None, image_data):
                    print('Invalid image, not loading')
                    continue
                
                foundimage = Image.open( io.BytesIO(image_data))
                break

            copy_p = copy.copy(p)            
            
            if (aspect_chk):
                
                aspect = foundimage.width / foundimage.height
                totalpixels = copy_p.width * copy_p.height
                
                aspect = (aspect + 1) / 2
                
                pixelwidth = copy_p.width
                
                if (aspect < 0.9 or aspect > 1.2):
                    copy_p.width = int(pixelwidth * aspect)
                    copy_p.width = int(round(pixelwidth / 64) * 64)
                    copy_p.height = int(pixelwidth * (1 / aspect))
                    copy_p.height = int(round(pixelwidth / 64) * 64)


            for k, v in args.items():
                setattr(copy_p, k, v)

            copy_p.init_images[0] = foundimage

            proc = process_images(copy_p)
            images += proc.images
            
            all_prompts += proc.all_prompts
            infotexts += proc.infotexts

        return Processed(p, images, p.seed, "", all_prompts=all_prompts, infotexts=infotexts)
