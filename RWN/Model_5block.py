import torch.nn as nn
from Utils import Node, get_graph_info, build_graph, save_graph, load_graph
import torch
import math
import os
import numpy as np



class depthwise_separable_conv_3(nn.Module):

    def __init__(self, nin, nout, stride):
        super(depthwise_separable_conv_3, self).__init__()
        #kernel size 
        #padding = kernel_size/2
        self.depthwise = nn.Conv1d(nin, nin, kernel_size=3, stride=stride, padding=1, groups=nin)
        self.pointwise = nn.Conv1d(nin, nout, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        out = self.pointwise(x)
        return out


class Triplet_unit(nn.Module):

    def __init__(self, inplanes, outplanes, stride=1):
        super(Triplet_unit, self).__init__()
        self.relu = nn.ReLU()
        self.conv = depthwise_separable_conv_3(inplanes, outplanes, stride)
        self.bn = nn.BatchNorm1d(outplanes)

    def forward(self, x):
        x = self.relu(x)
        x = self.conv(x)
        out = self.bn(x)
        return out


class Node_OP(nn.Module):

    def __init__(self, Node, inplanes, outplanes):
        super(Node_OP, self).__init__()
        self.is_input_node = Node.type == 0
        self.input_nums = len(Node.inputs)
        if self.input_nums > 1:
            self.mean_weight = nn.Parameter(torch.ones(self.input_nums))
            self.sigmoid = nn.Sigmoid()
        if self.is_input_node:
            self.conv = Triplet_unit(inplanes, outplanes, stride=2)
        else:
            self.conv = Triplet_unit(outplanes, outplanes, stride=1)

    def forward(self, *input):
      if self.input_nums > 1:
          out = self.sigmoid(self.mean_weight[0]) * input[0]
          for i in range(1,self.input_nums):
            #print(np.shape(input[i]))
            #print(i)
            out = out + self.sigmoid(self.mean_weight[i]) * input[i]
            
      else:
          out = input[0]
      out = self.conv(out)
      #print(np.shape(out))
      return out


class StageBlock(nn.Module):

    def __init__(self, graph, inplanes, outplanes):
        super(StageBlock, self).__init__()
        self.nodes, self.input_nodes, self.output_nodes = get_graph_info(graph)
        self.nodeop  = nn.ModuleList()
        for node in self.nodes:
            self.nodeop.append(Node_OP(node, inplanes, outplanes))

    def forward(self, x):
        results = {}
        for id in self.input_nodes:
            results[id] = self.nodeop[id](x)
        for id, node in enumerate(self.nodes):
            if id not in self.input_nodes:
                results[id] = self.nodeop[id](*[results[_id] for _id in node.inputs])
        result = results[self.output_nodes[0]]
        for idx, id in enumerate(self.output_nodes):
            if idx > 0:
                result = result + results[id]
        result = result / len(self.output_nodes)
        return result

class CNN(nn.Module):

    def __init__(self, args, num_classes=3):
        super(CNN, self).__init__()

        self.conv1 = depthwise_separable_conv_3(1, args.channels // 2, 2)
        self.bn1 = nn.BatchNorm1d(args.channels // 2)
	
        if args.resume:
            graph = load_graph(os.path.join(args.model_dir, 'conv2.yaml'))
        else:
            graph = build_graph(args.nodes // 5, args)
            save_graph(graph, os.path.join(args.model_dir, 'conv2.yaml'))
        self.conv2 = StageBlock(graph, args.channels // 2, args.channels)
    
        if args.resume:
            graph = load_graph(os.path.join(args.model_dir, 'conv3.yaml'))
        else:
            graph = build_graph(args.nodes//5, args)
            save_graph(graph, os.path.join(args.model_dir, 'conv3.yaml'))
        self.conv3 = StageBlock(graph, args.channels, args.channels )
  
  
        if args.resume:
            graph = load_graph(os.path.join(args.model_dir, 'conv4.yaml'))
        else:
            graph = build_graph(args.nodes//5, args)
            save_graph(graph, os.path.join(args.model_dir, 'conv4.yaml'))
        self.conv4 = StageBlock(graph, args.channels , args.channels)
  

        if args.resume:
            graph = load_graph(os.path.join(args.model_dir, 'conv5.yaml'))
        else:
            graph = build_graph(args.nodes//5, args)
            save_graph(graph, os.path.join(args.model_dir, 'conv5.yaml'))
        self.conv5 = StageBlock(graph, args.channels , args.channels)


        if args.resume:
            graph = load_graph(os.path.join(args.model_dir, 'conv6.yaml'))
        else:
            graph = build_graph(args.nodes//5, args)
            save_graph(graph, os.path.join(args.model_dir, 'conv6.yaml'))
        self.conv6 = StageBlock(graph, args.channels , args.channels)
       
  
        self.relu = nn.ReLU()
        self.conv = nn.Conv1d(args.channels , 32, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(32)
	  
        self.avgpool = nn.AvgPool1d(1, stride=1)
        self.gap = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(32, num_classes)
    

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


