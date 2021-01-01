import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])

# self.optimizer is for architecture. optimizer is for weights


class Architect(object):

    def __init__(self, model_1, model_2, c_lambda, args, criterion):
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model_1 = model_1
        self.model_2 = model_2
        self.criterion = criterion
        self.c_lambda = c_lambda
        self.args = args
        self.optimizer_1 = torch.optim.Adam(self.model_1.arch_parameters(),
                                            lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)
        self.optimizer_2 = torch.optim.Adam(self.model_2.arch_parameters(),
                                            lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)

    def _compute_unrolled_model(self, input, target, eta, network_optimizer_1, network_optimizer_2):
        loss = self.compute_loss(input, target)
        theta_1 = _concat(self.model_1.parameters()).data

        theta_2 = _concat(self.model_2.parameters()).data

        try:
            moment_1 = _concat(network_optimizer_1.state[v]['momentum_buffer']
                               for v in self.model_1.parameters()).mul_(self.network_momentum)
        except:
            moment_1 = torch.zeros_like(theta_1)
        dtheta_1 = _concat(torch.autograd.grad(
            loss, self.model_1.parameters(), retain_graph=True)).data + self.network_weight_decay*theta_1

        try:
            moment_2 = _concat(network_optimizer_2.state[v]['momentum_buffer']
                               for v in self.model_2.parameters()).mul_(self.network_momentum)
        except:
            moment_2 = torch.zeros_like(theta_2)

        dtheta_2 = _concat(torch.autograd.grad(
            loss, self.model_2.parameters())).data + self.network_weight_decay*theta_2

        unrolled_model_1, unrolled_model_2 = self._construct_model_from_theta(
            theta_1.sub(eta, moment_1+dtheta_1), theta_2.sub(eta, moment_2+dtheta_2))
        return unrolled_model_1, unrolled_model_2

    def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer_1, network_optimizer_2, unrolled):
        self.optimizer_1.zero_grad()
        self.optimizer_2.zero_grad()
        if unrolled:
            self._backward_step_unrolled(
                input_train, target_train, input_valid, target_valid, eta, network_optimizer_1, network_optimizer_2)
        else:
            self._backward_step(input_valid, target_valid)
        self.optimizer_1.step()
        self.optimizer_2.step()

    # def _backward_step(self, input_valid, target_valid):
    #     loss_1, _ = self.model_1._loss(input_valid, target_valid)
    #     loss_2, _ = self.model_2._loss(input_valid, target_valid)

    #     loss_1.backward()
    #     loss_2.backward()

    def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, lr, optimizer_1, optimizer_2):
        unrolled_model_1, unrolled_model_2 = self._compute_unrolled_model(
            input_train, target_train, lr, optimizer_1, optimizer_2)
        unrolled_loss_1, _ = unrolled_model_1._loss(input_valid, target_valid)
        unrolled_loss_2, _ = unrolled_model_2._loss(input_valid, target_valid)

        unrolled_loss = unrolled_loss_1 + unrolled_loss_2
        unrolled_loss.backward()

        vector_1 = [v.grad.data for v in unrolled_model_1.parameters()]
        vector_2 = [v.grad.data for v in unrolled_model_2.parameters()]

        implicit_grads_a1 = self._hessian_vector_product_model_1(
            vector_1, vector_2, input_train, target_train)
        implicit_grads_a2 = self._hessian_vector_product_model_2(
            vector_1, vector_2, input_train, target_train)

        dalpha_1 = [v.grad for v in unrolled_model_1.arch_parameters()]
        dalpha_2 = [v.grad for v in unrolled_model_2.arch_parameters()]

        for g, ig in zip(dalpha_1, implicit_grads_a1):
            g.data.sub_(lr, ig.data)

        for g, ig in zip(dalpha_2, implicit_grads_a2):
            g.data.sub_(lr, ig.data)

        for v, g in zip(self.model_1.arch_parameters(), dalpha_1):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

        for v, g in zip(self.model_2.arch_parameters(), dalpha_2):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

    def _construct_model_from_theta(self, theta_1, theta_2):
        model_new_1 = self.model_1.new()
        model_dict_1 = self.model_1.state_dict()

        model_new_2 = self.model_2.new()
        model_dict_2 = self.model_2.state_dict()

        params, offset = {}, 0
        for k, v in self.model_1.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta_1[offset: offset+v_length].view(v.size())
            offset += v_length

        assert offset == len(theta_1)
        model_dict_1.update(params)
        model_new_1.load_state_dict(model_dict_1)

        params, offset = {}, 0
        for k, v in self.model_2.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta_2[offset: offset+v_length].view(v.size())
            offset += v_length

        assert offset == len(theta_2)
        model_dict_2.update(params)
        model_new_2.load_state_dict(model_dict_2)

        return model_new_1.cuda(), model_new_2.cuda()

    # For model 1
    def _hessian_vector_product_model_1(self, vector_1, vector_2, input, target, r=1e-2):
        R = r / _concat(vector_1).norm()
        for p, v in zip(self.model_1.parameters(), vector_1):
            p.data.add_(R, v)
        loss = self.compute_loss(input, target)
        grads_p = torch.autograd.grad(
            loss, self.model_1.arch_parameters())

        for p, v in zip(self.model_1.parameters(), vector_1):
            p.data.sub_(2*R, v)
        loss = self.compute_loss(input, target)
        grads_n = torch.autograd.grad(
            loss, self.model_1.arch_parameters())

        for p, v in zip(self.model_1.parameters(), vector_1):
            p.data.add_(R, v)

        ig_1 = [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]

        R = r / _concat(vector_2).norm()
        for p, v in zip(self.model_2.parameters(), vector_2):
            p.data.add_(R, v)
        loss = self.compute_loss(input, target)
        grads_p = torch.autograd.grad(
            loss, self.model_1.arch_parameters())

        for p, v in zip(self.model_2.parameters(), vector_2):
            p.data.sub_(2*R, v)
        loss = self.compute_loss(input, target)
        grads_n = torch.autograd.grad(
            loss, self.model_1.arch_parameters())

        for p, v in zip(self.model_2.parameters(), vector_2):
            p.data.add_(R, v)

        ig_2 = [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]

        return [(x+y) for x, y in zip(ig_1, ig_2)]

    # For model 2
    def _hessian_vector_product_model_2(self, vector_1, vector_2, input, target, r=1e-2):
        R = r / _concat(vector_1).norm()
        for p, v in zip(self.model_1.parameters(), vector_1):
            p.data.add_(R, v)
        loss = self.compute_loss(input, target)
        grads_p = torch.autograd.grad(
            loss, self.model_2.arch_parameters())

        for p, v in zip(self.model_1.parameters(), vector_1):
            p.data.sub_(2*R, v)
        loss = self.compute_loss(input, target)
        grads_n = torch.autograd.grad(
            loss, self.model_2.arch_parameters())

        for p, v in zip(self.model_1.parameters(), vector_1):
            p.data.add_(R, v)

        ig_1 = [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]

        R = r / _concat(vector_2).norm()
        for p, v in zip(self.model_2.parameters(), vector_2):
            p.data.add_(R, v)
        loss = self.compute_loss(input, target)
        grads_p = torch.autograd.grad(
            loss, self.model_2.arch_parameters())

        for p, v in zip(self.model_2.parameters(), vector_2):
            p.data.sub_(2*R, v)
        loss = self.compute_loss(input, target)
        grads_n = torch.autograd.grad(
            loss, self.model_2.arch_parameters())

        for p, v in zip(self.model_2.parameters(), vector_2):
            p.data.add_(R, v)

        ig_2 = [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]

        return [(x+y) for x, y in zip(ig_1, ig_2)]

    def compute_loss(self, input, target):
        logits_1 = self.model_1(input)
        loss_1 = self.criterion(logits_1, target)

        logits_2 = self.model_2(input)
        loss_2 = self.criterion(logits_2, target)

        logits_1 = F.softmax(logits_1, dim=-1)
        logits_2 = F.softmax(logits_2, dim=-1)
        logits_3 = torch.log10(logits_1)
        logits_4 = torch.log10(logits_2)

        loss_add = torch.sum(logits_2*logits_3*-1) + \
            torch.sum(logits_1*logits_4*-1)

        loss = (loss_1 + loss_2) + self.args.c_lambda*(loss_add)
        return loss
