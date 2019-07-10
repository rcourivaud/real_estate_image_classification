import torch
import torch.distributed as dist
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.nn as nn
import torch.optim as optim
import sagemaker_containers
import argparse
import logging
import os
from real_estate_image_type.data import ImageFolder, transformations
from real_estate_image_type.model import build_model, save_model

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def untar_data(directory):
    logger.debug(f"Start to untar data in directory {directory}")

    fname = 'data.tar.gz'
    path = os.path.join(directory, fname)
    logger.debug(f"Absolute path {path}")

    os.system(f'!tar -xzf {path} -C {directory} && rm {path}')
    logger.debug(f"Successfully extract data")


def __train_model(model,train_loader,valid_loader, epochs,  criterion, optimizer, device):
    for epoch in range(epochs):
        train_loss = 0
        val_loss = 0
        accuracy = 0

        logger.debug(f"Start training...")
        model.train()
        counter = 0
        logger.debug(f" -> TRAINING EPOCH {epoch}")
        for inputs, labels in train_loader:

            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*inputs.size(0)
            counter +=1

        model.eval()
        logger.debug(f" -> VALIDATING EPOCH {epoch}")

        with torch.no_grad():
            for inputs, labels in tqdm(valid_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                output = model.forward(inputs)
                valloss = criterion(output, labels)
                val_loss += valloss.item()*inputs.size(0)

                output = torch.exp(output)
                top_p, top_class = output.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.sum(equals.type(torch.FloatTensor)).item()

        # Get the average loss for the entire epoch
        train_loss = train_loss/(train_loader.batch_size*(counter))
        valid_loss = val_loss/(valid_loader.batch_size*(counter))
        # Print out the information
        acc = accuracy/(valid_loader.batch_size*counter)
        logger.debug('Accuracy: ', acc)
        logger.debug('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))

    return model


def _train(args):
    is_distributed = len(args.hosts) > 1 and args.dist_backend is not None
    logger.debug("Distributed training - {}".format(is_distributed))

    if is_distributed:
        # Initialize the distributed environment.
        world_size = len(args.hosts)
        os.environ['WORLD_SIZE'] = str(world_size)
        host_rank = args.hosts.index(args.current_host)
        os.environ['RANK'] = str(host_rank)
        dist.init_process_group(backend=args.dist_backend, rank=host_rank, world_size=world_size)
        logger.info(
            'Initialized the distributed environment: \'{}\' backend on {} nodes. '.format(
                args.dist_backend,
                dist.get_world_size()) + 'Current host rank is {}. Using cuda: {}. Number of gpus: {}'.format(
                dist.get_rank(), torch.cuda.is_available(), args.num_gpus))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info("Device Type: {}".format(device))

    logger.info("Loading dataset...")
    train_set = ImageFolder(args.data_dir, transform = transformations )
    valid_set = ImageFolder(args.data_dir, transform = transformations)

    logger.info("Loading data loader...")
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=args.batch_size, shuffle=True)

    num_labels = len(train_set.classes)
    logger.info("Data was loaded and {num_labels} were detected".format(num_labels=num_labels))
    logger.info("Start downloading DenseNet161 weights for TL")

    model = build_model(num_labels, device)

    criterion = nn.NLLLoss().to(device)
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)

    model = __train_model(model, train_loader, valid_loader,args.epochs, criterion, optimizer, device)

    return save_model(model, args.model_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--workers', type=int, default=2, metavar='W',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', type=int, default=5, metavar='E',
                        help='number of total epochs to run (default: 2)')
    parser.add_argument('--batch_size', type=int, default=16, metavar='BS',
                        help='batch size (default: 4)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='initial learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--dist_backend', type=str, default='gloo', help='distributed backend (default: gloo)')

    env = sagemaker_containers.training_env()
    parser.add_argument('--hosts', type=list, default=env.hosts)
    parser.add_argument('--current-host', type=str, default=env.current_host)
    parser.add_argument('--model-dir', type=str, default=env.model_dir)
    parser.add_argument('--data-dir', type=str, default=env.channel_input_dirs.get('training'))
    parser.add_argument('--num-gpus', type=int, default=env.num_gpus)

    args = parser.parse_args()
    untar_data(args.data_dir)
    _train(args)
