use clap::Parser;
use std::cmp::min;
use std::fs::File;
use std::io::{BufReader, Read, Write};
use tqdm::pbar;
use zip2zip_compression::{codec, config};

#[derive(Parser, Debug)]
struct Args {
    #[arg(short, long, default_value = "fineweb10B")]
    name: String,
    #[arg(long, default_value = "1")]
    num_chunks: usize,
    #[arg(long, default_value = "50257")]
    initial_vocab_size: usize,
    #[arg(long, default_value = "1024")]
    max_codebook_size: usize,
    #[arg(long, default_value = "4")]
    max_subtokens: usize,
    #[arg(long, default_value = "1024")]
    max_out_seq_length: usize,
    #[arg(long, default_value = "50256")]
    eot_token_id: usize,
}

fn convert_codebook_to_vec(codebook: &codec::Codebook) -> Vec<usize> {
    let config = codebook.config.clone();
    let total_size = config.max_codebook_size * config.max_subtokens;
    let mut output = vec![config.pad_token_id; total_size];

    let mut entries: Vec<_> = codebook.inner.iter().collect();
    entries.sort_by_key(|(_, id)| *id);

    for (index, (ids, _)) in entries.iter().enumerate() {
        if index >= config.max_codebook_size {
            break;
        }

        let start_idx = index * config.max_subtokens;
        let end_idx = start_idx + ids.len().min(config.max_subtokens);
        output[start_idx..end_idx].copy_from_slice(&ids[..ids.len().min(config.max_subtokens)]);
    }

    output
}

fn compress_file(filename: &str, args: &Args) {
    let file = File::open(format!("../{}/{}", args.name, filename)).unwrap();
    let mut reader = BufReader::new(file);

    let mut header_buffer = vec![0u8; 256 * 4];
    reader.read_exact(&mut header_buffer).unwrap();
    let mut header: Vec<i32> = header_buffer
        .chunks_exact(4)
        .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    let mut ids_buffer = Vec::new();
    reader.read_to_end(&mut ids_buffer).unwrap();

    let ids: Vec<usize> = ids_buffer
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]) as usize)
        .collect();

    assert!(
        header[0] == 20240520,
        "magic number mismatch in the data .bin file"
    );
    assert!(header[1] == 1, "unsupported version");
    let num_tokens = header[2] as usize;

    let compression_config = config::CompressionConfig::new(
        args.initial_vocab_size,
        args.max_codebook_size,
        args.max_subtokens,
        args.eot_token_id,
        Some(vec![args.eot_token_id]),
    );

    let mut compressed_ids: Vec<usize> = Vec::new();
    let mut codebook_vec: Vec<usize> = Vec::new();

    let mut i: usize = 0;
    let mut pb = pbar(Some(num_tokens));
    while i < num_tokens && (num_tokens - i) > args.max_out_seq_length {
        let mut compression_state = codec::CompressionState::new(compression_config.clone());

        let (c_ids, remaining_ids_offset) = codec::encode_fn(
            &mut compression_state,
            &ids,
            i,
            config::PaddingStrategy::DoNotPad,
            true,
            Some(args.max_out_seq_length),
        );

        let _ = pb.update(min(remaining_ids_offset - i, num_tokens - i));
        i = remaining_ids_offset;

        if c_ids.len() != args.max_out_seq_length {
            println!("c_ids.len(): {}", c_ids.len());
            return;
        }

        compressed_ids.extend(c_ids);
        codebook_vec.extend(convert_codebook_to_vec(&compression_state.codebook));
    }
    let _ = pb.close();

    println!("compressed_ids.len(): {}", compressed_ids.len());
    println!("codebook_vec.len(): {}", codebook_vec.len());

    header[2] = compressed_ids.len() as i32;
    header[3] = (codebook_vec.len() / (args.max_codebook_size * args.max_subtokens)) as i32;
    header[4] = args.max_codebook_size as i32;
    header[5] = args.max_subtokens as i32;

    let mut compressed_file =
        File::create(format!("../{}/compressed_{}", args.name, filename)).unwrap();
    let header_bytes: Vec<u8> = header.iter().flat_map(|&x| x.to_le_bytes()).collect();
    compressed_file.write_all(&header_bytes).unwrap();
    let compressed_ids_bytes: Vec<u8> = compressed_ids
        .iter()
        .flat_map(|&x| (x as u16).to_le_bytes())
        .collect();
    compressed_file.write_all(&compressed_ids_bytes).unwrap();

    let mut codebook_file =
        File::create(format!("../{}/codebooks_{}", args.name, filename)).unwrap();
    let codebook_bytes: Vec<u8> = codebook_vec
        .iter()
        .flat_map(|&x| (x as u16).to_le_bytes())
        .collect();
    codebook_file.write_all(&codebook_bytes).unwrap();
}

fn main() {
    let args = Args::parse();

    let mut filename = format!("fineweb_val_{:06}.bin", 0);
    compress_file(&filename, &args);

    for chunk in 1..args.num_chunks + 1 {
        filename = format!("fineweb_train_{:06}.bin", chunk);
        compress_file(&filename, &args);
    }
}
