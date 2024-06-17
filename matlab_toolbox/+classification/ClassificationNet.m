classdef ClassificationNet<handle
    properties (Constant)
        supported_task={'binary' 'drone-classification'};
        supported_arch={'resnet18' 'resnet50' 'efficientnet_v2_s' 'efficientnet_v2_m' 'resnext50_32x4d' 'alexnet'}; %'convnext_tiny' 'convnext_base'
        input_size=[224 224 3];
        num2class_binary = dictionary([1, 2],{'non-drone', 'drone'});
        num2class_drone_classification = dictionary([1, 2, 3, 4, 5], {'Autel_Evo_II' 'DJI_Matrice_210' 'DJI_Mavic_3' 'DJI_Mini_2' 'Yuneec_H520E'});
    end
    properties (SetAccess=private)
        net;
        arch;
        task;
    end
    methods
        function obj=ClassificationNet(arch,task)
            assert(ismember(arch, obj.supported_arch), 'Unsupported architecture');
            assert(ismember(task, obj.supported_task), 'Unsupported task');
            obj.net = importNetworkFromPyTorch(fullfile(fileparts(mfilename('fullpath')),'pt_models',[task '-' arch '.pt']),PyTorchInputSizes=[NaN,3,224,224]);
            obj.arch = arch;
            obj.task = task;
        end

        function img_dlarray=transforms(obj,img)
            img = imresize(img, obj.input_size(1:2));
            img = rescale(img, 0, 1);
            img_dlarray = dlarray(single(img),'SSCB');
        end

        function class_label=predict_label(obj,img)
            img_dlarray = obj.transforms(img);
            prob = predict(obj.net, img_dlarray);
            [~,label_ind] = max(prob);
            if strcmp(obj.task, 'binary')
                class_label = obj.num2class_binary(label_ind);
            elseif strcmp(obj.task, 'drone-classification')
                class_label = obj.num2class_drone_classification(label_ind);
            else
                error('Not supported');
            end
            class_label=class_label{1};
        end
    end
end
