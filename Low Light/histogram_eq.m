clear;
clc;    

input_folder  = 'low light';
output_folder = 'he color';
extensions = {'*.jpg','*.jpeg','*.png','*.bmp'};

for k = 1:length(extensions)
    files = dir(fullfile(input_folder, extensions{k}));

    for i = 1:length(files)
        filename = files(i).name;
        infile = fullfile(input_folder, filename);

        I = imread(infile);

        if ndims(I) == 3
            lab = rgb2lab(I);
            L = lab(:,:,1);
            L_uint8 = uint8(255 * (L / 100));

            L_eq_uint8 = histo_eq(L_uint8);
        
            L_eq = double(L_eq_uint8) / 255 * 100;
        
            lab(:,:,1) = L_eq;
        
            I_eq = lab2rgb(lab);
            I_eq = im2uint8(I_eq);
        else
            I_eq = histo_eq(I);
        end

        outfile = fullfile(output_folder, filename);
        imwrite(I_eq, outfile);

        fprintf('Processed: %s → %s\n', infile, outfile);
    end
end

function I_eq = histo_eq(I)

    I = uint8(I);

    pdf = accumarray(double(I(:)) + 1, 1, [256 1]);

    cdf = cumsum(pdf) / numel(I);

    cdf_scaled = uint8(255 * cdf);

    I_eq = reshape(cdf_scaled(double(I(:)) + 1), size(I));
end

