local ROTSizeCriterion_boundary, parent = torch.class('nn.ROTSizeCriterion_boundary', 'nn.Criterion')

function ROTSizeCriterion_boundary:__init()
   parent.__init(self)

   self.p1 = -1/3
   self.q1 = -1/3
   self.p2 = 1/3
   self.q2 = -1/3
   self.p3 = -1/3
   self.q3 = 1/3
   self.p4 = 1/3
   self.q4 = 1/3
   self.lambda_b = 0.55
   self.lambda_ob = 0.25
end

function ROTSizeCriterion_boundary:updateOutput(transformParams, gtLabels)

   -- Compute normalized (-1 ~ +1) gtCenter
   -- gtCenter is y*224+x 
   labels_y = torch.div(gtLabels, 224):floor()
   labels_x = gtLabels - (labels_y * 224)
   norm_labels_x = (labels_x - 112) / 112
   norm_labels_y = (labels_y - 112) / 112
   norm_gtCenter_x = norm_labels_x:select(4,1):select(3,1):select(2,1)
   norm_gtCenter_y = norm_labels_y:select(4,1):select(3,1):select(2,1)
   norm_gtObject_x1 = norm_labels_x:select(4,2):select(3,1):select(2,1)
   norm_gtObject_y1 = norm_labels_y:select(4,2):select(3,1):select(2,1)
   norm_gtObject_x2 = norm_labels_x:select(4,3):select(3,1):select(2,1)
   norm_gtObject_y2 = norm_labels_y:select(4,3):select(3,1):select(2,1)


   -- Assume that shape of transformParams is like that....
   s = transformParams:select(2,1)
   tx = transformParams:select(2,2)
   ty = transformParams:select(2,3)

   n = torch.numel(s)

   -- transformed image boundary
   b_left = (-1 - tx):cdiv(s)
   b_top = (-1 - ty):cdiv(s)
   b_bottom = (1 - ty):cdiv(s)
   b_right = (1 - tx):cdiv(s)

   -- object boundary
   ob_left = (norm_gtObject_x1 - tx):cdiv(s)
   ob_top = (norm_gtObject_y1 - ty):cdiv(s)
   ob_bottom = (norm_gtObject_y2 - ty):cdiv(s)
   ob_right = (norm_gtObject_x2 - tx):cdiv(s)

   -- object_ratio
   self.ob_ratio = (norm_gtObject_x2 - norm_gtObject_x1):cdiv(s):cmul((norm_gtObject_y2 - norm_gtObject_y1):cdiv(s)) / 4




   -- black region boundary dist 
   self.dist_left = torch.Tensor(gtLabels:size()[1], 1):fill(0)
   self.dist_top = torch.Tensor(gtLabels:size()[1], 1):fill(0)
   self.dist_bottom = torch.Tensor(gtLabels:size()[1], 1):fill(0)
   self.dist_right = torch.Tensor(gtLabels:size()[1], 1):fill(0)

   -- object dist 
   self.dist_left_ob = torch.Tensor(gtLabels:size()[1], 1):fill(0)
   self.dist_top_ob = torch.Tensor(gtLabels:size()[1], 1):fill(0)
   self.dist_bottom_ob = torch.Tensor(gtLabels:size()[1], 1):fill(0)
   self.dist_right_ob = torch.Tensor(gtLabels:size()[1], 1):fill(0)
   self.dist_ob_ratio = torch.Tensor(gtLabels:size()[1], 1):fill(0)


   for i = 1, n do
      -- black region boundary constraint
      if b_left[i] <= -1 then
         self.dist_left[i] = 0
      else
         self.dist_left[i] = (b_left[i] - (-1))^2
      end
      if b_top[i] <= -1 then
         self.dist_top[i] = 0
      else
         self.dist_top[i] = (b_top[i] - (-1))^2
      end
      if b_bottom[i] >= 1 then
         self.dist_bottom[i] = 0
      else
         self.dist_bottom[i] = (b_bottom[i] - (1))^2
      end
      if b_right[i] >= 1 then
         self.dist_right[i] = 0
      else
         self.dist_right[i] = (b_right[i] - (1))^2
      end

      -- object inside image constraint
      if ob_left[i] > -1 then
         self.dist_left_ob[i] = 0
      else
         self.dist_left_ob[i] = (ob_left[i] - (-1))^2
      end
      if ob_top[i] > -1 then
         self.dist_top_ob[i] = 0
      else
         self.dist_top_ob[i] = (ob_top[i] - (-1))^2
      end
      if ob_bottom[i] < 1 then
         self.dist_bottom_ob[i] = 0
      else
         self.dist_bottom_ob[i] = (ob_bottom[i] - (1))^2
      end
      if ob_right[i] < 1 then
         self.dist_right_ob[i] = 0
      else
         self.dist_right_ob[i] = (ob_right[i] - (1))^2
      end

      -- object ratio constraint
      if self.ob_ratio[i] < 0.1 then
         self.dist_ob_ratio[i] = (self.ob_ratio[i] - (0.1))^2
      end
      if self.ob_ratio[i] > 0.8 then
         self.dist_ob_ratio[i] = (self.ob_ratio[i] - (0.8))^2
      end

   end

   dist1 = ((norm_gtCenter_x-tx):cdiv(s) - self.p1):pow(2) + ((norm_gtCenter_y-ty):cdiv(s) - self.q1):pow(2)
   dist2 = ((norm_gtCenter_x-tx):cdiv(s) - self.p2):pow(2) + ((norm_gtCenter_y-ty):cdiv(s) - self.q2):pow(2)
   dist3 = ((norm_gtCenter_x-tx):cdiv(s) - self.p3):pow(2) + ((norm_gtCenter_y-ty):cdiv(s) - self.q3):pow(2)
   dist4 = ((norm_gtCenter_x-tx):cdiv(s) - self.p4):pow(2) + ((norm_gtCenter_y-ty):cdiv(s) - self.q4):pow(2) 


   -- dist1 = ((norm_gtCenter_x-tx):cdiv(s) - self.p1):pow(2) + ((norm_gtCenter_y-ty):cdiv(s) - self.q1):pow(2) + self.lambda * s:abs()
   -- dist2 = ((norm_gtCenter_x-tx):cdiv(s) - self.p2):pow(2) + ((norm_gtCenter_y-ty):cdiv(s) - self.q2):pow(2) + self.lambda * s:abs()
   -- dist3 = ((norm_gtCenter_x-tx):cdiv(s) - self.p3):pow(2) + ((norm_gtCenter_y-ty):cdiv(s) - self.q3):pow(2) + self.lambda * s:abs()
   -- dist4 = ((norm_gtCenter_x-tx):cdiv(s) - self.p4):pow(2) + ((norm_gtCenter_y-ty):cdiv(s) - self.q4):pow(2) + self.lambda * s:abs()

   cat_dist = torch.cat({dist1, dist2, dist3, dist4}, 2)
   min_dist, self.indices = torch.min(cat_dist, 2)
   min_dist = min_dist + self.lambda_b * (self.dist_left + self.dist_top + self.dist_bottom + self.dist_right) + self.lambda_ob * (self.dist_left_ob + self.dist_top_ob + self.dist_bottom_ob + self.dist_right_ob + self.dist_ob_ratio)

   return min_dist
end


function ROTSizeCriterion_boundary:updateGradInput(transformParams, gtLabels)
   
   -- self.gradInput:resizeAs(transformParams)
   -- Compute normalized (-1 ~ +1) gtCenter
   -- gtCenter is y*224+x 
   labels_y = torch.div(gtLabels, 224):floor()
   labels_x = gtLabels - (labels_y * 224)
   norm_labels_x = (labels_x - 112) / 112
   norm_labels_y = (labels_y - 112) / 112
   norm_gtCenter_x = norm_labels_x:narrow(4, 1, 1)
   norm_gtCenter_y = norm_labels_y:narrow(4, 1, 1)
   norm_gtObject_x1 = norm_labels_x:narrow(4, 2, 1)
   norm_gtObject_y1 = norm_labels_y:narrow(4, 2, 1)
   norm_gtObject_x2 = norm_labels_x:narrow(4, 3, 1)
   norm_gtObject_y2 = norm_labels_y:narrow(4, 3, 1)

   -- Assume that shape of transformParams is like that....
   s = transformParams:select(2,1)
   tx = transformParams:select(2,2)
   ty = transformParams:select(2,3)


   grad_s = torch.Tensor(gtLabels:size()[1], 1)
   grad_tx = torch.Tensor(gtLabels:size()[1], 1)
   grad_ty = torch.Tensor(gtLabels:size()[1], 1)

   -- grad of black region boundary term
   grad_left_s = torch.Tensor(gtLabels:size()[1], 1):fill(0)
   grad_left_tx = torch.Tensor(gtLabels:size()[1], 1):fill(0)
   grad_top_s = torch.Tensor(gtLabels:size()[1], 1):fill(0)
   grad_top_ty = torch.Tensor(gtLabels:size()[1], 1):fill(0)
   grad_bottom_s = torch.Tensor(gtLabels:size()[1], 1):fill(0)
   grad_bottom_ty = torch.Tensor(gtLabels:size()[1], 1):fill(0)
   grad_right_s = torch.Tensor(gtLabels:size()[1], 1):fill(0)
   grad_right_tx = torch.Tensor(gtLabels:size()[1], 1):fill(0)

   -- grad of object inside image term
   grad_left_ob_s = torch.Tensor(gtLabels:size()[1], 1):fill(0)
   grad_left_ob_tx = torch.Tensor(gtLabels:size()[1], 1):fill(0)
   grad_top_ob_s = torch.Tensor(gtLabels:size()[1], 1):fill(0)
   grad_top_ob_ty = torch.Tensor(gtLabels:size()[1], 1):fill(0)
   grad_bottom_ob_s = torch.Tensor(gtLabels:size()[1], 1):fill(0)
   grad_bottom_ob_ty = torch.Tensor(gtLabels:size()[1], 1):fill(0)
   grad_right_ob_s = torch.Tensor(gtLabels:size()[1], 1):fill(0)
   grad_right_ob_tx = torch.Tensor(gtLabels:size()[1], 1):fill(0)

   -- grad of object ratio term
   grad_ob_ratio_s = torch.Tensor(gtLabels:size()[1], 1):fill(0)

   for i = 1, self.indices:size()[1] do
      -- compute black region boundary gradient
      if self.dist_left[i][1] > 0 then
         grad_left_s[i] = 2 * (((-1) - tx[i])/(s[i]) - (-1)) * ((1 + tx[i])/(s[i]^2))
         grad_left_tx[i] = 2 * (((-1) - tx[i])/(s[i]) - (-1)) * ((-1) / s[i])
      end
      if self.dist_top[i][1] > 0 then
         grad_top_s[i] = 2 * (((-1) - ty[i])/(s[i]) - (-1)) * ((1 + ty[i])/(s[i]^2))
         grad_top_ty[i] = 2 * (((-1) - ty[i])/(s[i]) - (-1)) * ((-1) / s[i])
      end
      if self.dist_bottom[i][1] > 0 then
         grad_bottom_s[i] = 2 * (((1) - ty[i])/(s[i]) - (1)) * (((-1) + ty[i])/(s[i]^2))
         grad_bottom_ty[i] = 2 * (((1) - ty[i])/(s[i]) - (1)) * ((-1) / s[i])
      end
      if self.dist_right[i][1] > 0 then
         grad_right_s[i] = 2 * (((1) - tx[i])/(s[i]) - (1)) * (((-1) + tx[i])/(s[i]^2))
         grad_right_tx[i] = 2 * (((1) - tx[i])/(s[i]) - (1)) * ((-1) / s[i])
      end

      -- compute object inside image gradient
      if self.dist_left_ob[i][1] > 0 then
         grad_left_ob_s[i] = 2 * ((norm_gtObject_x1[i] - tx[i])/(s[i]) - (-1)):cmul((-1)*((norm_gtObject_x1[i] - tx[i])/(s[i]^2)))
         grad_left_ob_tx[i] = 2 * ((norm_gtObject_x1[i] - tx[i])/(s[i]) - (-1)) * ((-1) / s[i])
      end
      if self.dist_top_ob[i][1] > 0 then
         grad_top_ob_s[i] = 2 * ((norm_gtObject_y1[i] - ty[i])/(s[i]) - (-1)):cmul((-1)*((norm_gtObject_y1[i] - ty[i])/(s[i]^2)))
         grad_top_ob_ty[i] = 2 * ((norm_gtObject_y1[i] - ty[i])/(s[i]) - (-1)) * ((-1) / s[i])
      end
      if self.dist_bottom_ob[i][1] > 0 then
         grad_bottom_ob_s[i] = 2 * ((norm_gtObject_y2[i] - ty[i])/(s[i]) - (1)):cmul((-1)*((norm_gtObject_y2[i] - ty[i])/(s[i]^2)))
         grad_bottom_ob_ty[i] = 2 * ((norm_gtObject_y2[i] - ty[i])/(s[i]) - (1)) * ((-1) / s[i])
      end
      if self.dist_right_ob[i][1] > 0 then
         grad_right_ob_s[i] = 2 * ((norm_gtObject_x2[i] - tx[i])/(s[i]) - (1)):cmul((-1)*((norm_gtObject_x2[i] - tx[i])/(s[i]^2)))
         grad_right_ob_tx[i] = 2 * ((norm_gtObject_x2[i] - tx[i])/(s[i]) - (1)) * ((-1) / s[i])
      end

      -- compute object ratio gradient
      if (self.dist_ob_ratio[i][1] > 0) and (self.ob_ratio[i] < 0.1) then
         grad_ob_ratio_s[i] = 2 * ((norm_gtObject_x2[i]-norm_gtObject_x1[i]):cmul((norm_gtObject_y2[i]-norm_gtObject_y1[i]))/(4*s[i]^2) - (0.1)):cmul((-0.5)*((norm_gtObject_x2[i]-norm_gtObject_x1[i]):cmul((norm_gtObject_y2[i]-norm_gtObject_y1[i]))/(s[i]^3)))
      end
      if (self.dist_ob_ratio[i][1] > 0) and (self.ob_ratio[i] > 0.8) then
         grad_ob_ratio_s[i] = 2 * ((norm_gtObject_x2[i]-norm_gtObject_x1[i]):cmul((norm_gtObject_y2[i]-norm_gtObject_y1[i]))/(4*s[i]^2) - (0.8)):cmul((-0.5)*((norm_gtObject_x2[i]-norm_gtObject_x1[i]):cmul((norm_gtObject_y2[i]-norm_gtObject_y1[i]))/(s[i]^3)))
      end

      -- compute final gradient
      if self.indices[i][1] == 1 then
         grad_s[i] = 2 * ((norm_gtCenter_x[i] - tx[i])/(s[i]) - self.p1):cmul((((-1) * (norm_gtCenter_x[i] - tx[i]))/(s[i]^2))) + 2 * ((norm_gtCenter_y[i] - ty[i])/(s[i]) - self.q1):cmul((((-1) * (norm_gtCenter_y[i] - ty[i]))/(s[i]^2))) 
         grad_tx[i] = 2 * ((norm_gtCenter_x[i] - tx[i])/(s[i]) - self.p1) * ((-1) / s[i]) 
         grad_ty[i] = 2 * ((norm_gtCenter_y[i] - ty[i])/(s[i]) - self.q1) * ((-1) / s[i]) 
      elseif self.indices[i][1] == 2 then
         grad_s[i] = 2 * ((norm_gtCenter_x[i] - tx[i])/(s[i]) - self.p2):cmul((((-1) * (norm_gtCenter_x[i] - tx[i]))/(s[i]^2))) + 2 * ((norm_gtCenter_y[i] - ty[i])/(s[i]) - self.q2):cmul((((-1) * (norm_gtCenter_y[i] - ty[i]))/(s[i]^2))) 
         grad_tx[i] = 2 * ((norm_gtCenter_x[i] - tx[i])/(s[i]) - self.p2) * ((-1) / s[i]) 
         grad_ty[i] = 2 * ((norm_gtCenter_y[i] - ty[i])/(s[i]) - self.q2) * ((-1) / s[i]) 

      elseif self.indices[i][1] == 3 then 
         grad_s[i] = 2 * ((norm_gtCenter_x[i] - tx[i])/(s[i]) - self.p3):cmul((((-1) * (norm_gtCenter_x[i] - tx[i]))/(s[i]^2))) + 2 * ((norm_gtCenter_y[i] - ty[i])/(s[i]) - self.q3):cmul((((-1) * (norm_gtCenter_y[i] - ty[i]))/(s[i]^2))) 
         grad_tx[i] = 2 * ((norm_gtCenter_x[i] - tx[i])/(s[i]) - self.p3) * ((-1) / s[i]) 
         grad_ty[i] = 2 * ((norm_gtCenter_y[i] - ty[i])/(s[i]) - self.q3) * ((-1) / s[i])

      elseif self.indices[i][1] == 4 then 
         grad_s[i] = 2 * ((norm_gtCenter_x[i] - tx[i])/(s[i]) - self.p4):cmul((((-1) * (norm_gtCenter_x[i] - tx[i]))/(s[i]^2))) + 2 * ((norm_gtCenter_y[i] - ty[i])/(s[i]) - self.q4):cmul((((-1) * (norm_gtCenter_y[i] - ty[i]))/(s[i]^2))) 
         grad_tx[i] = 2 * ((norm_gtCenter_x[i] - tx[i])/(s[i]) - self.p4) * ((-1) / s[i]) 
         grad_ty[i] = 2 * ((norm_gtCenter_y[i] - ty[i])/(s[i]) - self.q4) * ((-1) / s[i])
      end

      grad_s[i] = grad_s[i] + self.lambda_b * (grad_left_s[i] + grad_top_s[i] + grad_bottom_s[i] + grad_right_s[i]) + self.lambda_ob * (grad_left_ob_s[i] + grad_top_ob_s[i] + grad_bottom_ob_s[i] + grad_right_ob_s[i] + grad_ob_ratio_s[i])
      grad_tx[i] = grad_tx[i] + self.lambda_b * (grad_left_tx[i] + grad_right_tx[i]) + self.lambda_ob * (grad_left_ob_tx[i] + grad_right_ob_tx[i])
      grad_ty[i] = grad_ty[i] + self.lambda_b * (grad_top_ty[i] + grad_bottom_ty[i]) + self.lambda_ob * (grad_top_ob_ty[i] + grad_bottom_ob_ty[i])

   end


   -- for i = 1, self.indices:size()[1] do
   --    if s[i] > 0 then
   --       grad = 1
   --    elseif s[i] < 0 then
   --       grad = -1
   --    else
   --       grad = 0         
   --    end
   --    if self.indices[i][1] == 1 then
   --       grad_s[i] = 2 * ((norm_gtCenter_x[i] - tx[i])/(s[i]) - self.p1):cmul((((-1) * (norm_gtCenter_x[i] - tx[i]))/(s[i]^2))) + 2 * ((norm_gtCenter_y[i] - ty[i])/(s[i]) - self.q1):cmul((((-1) * (norm_gtCenter_y[i] - ty[i]))/(s[i]^2))) + self.lambda * grad
   --       grad_tx[i] = 2 * ((norm_gtCenter_x[i] - tx[i])/(s[i]) - self.p1) * ((-1) / s[i])
   --       grad_ty[i] = 2 * ((norm_gtCenter_y[i] - ty[i])/(s[i]) - self.q1) * ((-1) / s[i])
   --    elseif self.indices[i][1] == 2 then
   --       grad_s[i] = 2 * ((norm_gtCenter_x[i] - tx[i])/(s[i]) - self.p2):cmul((((-1) * (norm_gtCenter_x[i] - tx[i]))/(s[i]^2))) + 2 * ((norm_gtCenter_y[i] - ty[i])/(s[i]) - self.q2):cmul((((-1) * (norm_gtCenter_y[i] - ty[i]))/(s[i]^2))) + self.lambda * grad
   --       grad_tx[i] = 2 * ((norm_gtCenter_x[i] - tx[i])/(s[i]) - self.p2) * ((-1) / s[i])
   --       grad_ty[i] = 2 * ((norm_gtCenter_y[i] - ty[i])/(s[i]) - self.q2) * ((-1) / s[i])

   --    elseif self.indices[i][1] == 3 then 
   --       grad_s[i] = 2 * ((norm_gtCenter_x[i] - tx[i])/(s[i]) - self.p3):cmul((((-1) * (norm_gtCenter_x[i] - tx[i]))/(s[i]^2))) + 2 * ((norm_gtCenter_y[i] - ty[i])/(s[i]) - self.q3):cmul((((-1) * (norm_gtCenter_y[i] - ty[i]))/(s[i]^2))) + self.lambda * grad
   --       grad_tx[i] = 2 * ((norm_gtCenter_x[i] - tx[i])/(s[i]) - self.p3) * ((-1) / s[i])
   --       grad_ty[i] = 2 * ((norm_gtCenter_y[i] - ty[i])/(s[i]) - self.q3) * ((-1) / s[i])

   --    elseif self.indices[i][1] == 4 then 
   --       grad_s[i] = 2 * ((norm_gtCenter_x[i] - tx[i])/(s[i]) - self.p4):cmul((((-1) * (norm_gtCenter_x[i] - tx[i]))/(s[i]^2))) + 2 * ((norm_gtCenter_y[i] - ty[i])/(s[i]) - self.q4):cmul((((-1) * (norm_gtCenter_y[i] - ty[i]))/(s[i]^2))) + self.lambda * grad
   --       grad_tx[i] = 2 * ((norm_gtCenter_x[i] - tx[i])/(s[i]) - self.p4) * ((-1) / s[i])
   --       grad_ty[i] = 2 * ((norm_gtCenter_y[i] - ty[i])/(s[i]) - self.q4) * ((-1) / s[i])
   --    end

   -- end
   self.gradInput = torch.cat({grad_s, grad_tx, grad_ty}, 2)


   return self.gradInput
end
