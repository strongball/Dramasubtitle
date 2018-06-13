from dataset.readVideo import DramaDataset
import torchvision.transforms as transforms
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", help="Dataset dir video in dir named video", required=True)

transform = transforms.Compose([transforms.Resize(256),
                                ])

if __name__ == "__main__":
    args = parser.parse_args()
    
    datasets = DramaDataset(basedir=args.data,
                            #startSeries=2,
                            #maxSeries=1,
                            maxFrame=1,
                            timeOffset=0.2,
                            transform=transform,
                            useBmp=False
                           )
    for ep in datasets.epochs:
        epdir = os.path.join(datasets.frameDir, ep.order)
        if not os.path.isdir(epdir):
            os.mkdir(epdir)
        print("Make: {}".format(ep.order))
        count = 0
        for i, data in ep.data.iterrows():
            filename = os.path.join(epdir, str(data["start"])+".bmp")
            imgs, sucess = ep.getFrames(data["start"], data["end"])
            if sucess and len(imgs) > 0:
                imgs[0].save(filename)
                count += 1
            else:
                print("Error")
        print("Create: {}\n".format(count))

