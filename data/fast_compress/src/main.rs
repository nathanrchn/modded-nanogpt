use clap::Parser;
use fastset::Set;
use itertools::Itertools;
use rustc_hash::FxHashMap;
use std::cmp::min;
use std::fs::File;
use std::io::{BufReader, Read, Write};
use tqdm::pbar;

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

#[inline(always)]
fn codebook_contains(
    codebook: &FxHashMap<Vec<usize>, usize>,
    ids: &Vec<usize>,
    initial_vocab_size: usize,
) -> bool {
    if ids.len() == 1 {
        ids[0] < initial_vocab_size
    } else {
        codebook.contains_key(ids)
    }
}

#[inline(always)]
fn get_usize_from_codebook(codebook: &FxHashMap<Vec<usize>, usize>, ids: &Vec<usize>) -> usize {
    if ids.len() == 1 {
        ids[0]
    } else {
        codebook.get(ids).unwrap().clone()
    }
}

#[inline(always)]
fn disabled_ids_to_set(disabled_ids: Option<Vec<usize>>) -> Set {
    disabled_ids.map_or_else(
        || Set::with_capacity(0),
        |d_ids| {
            let mut set = Set::with_capacity(d_ids.iter().max().unwrap() + 1);
            for id in d_ids {
                set.insert(id);
            }
            set
        },
    )
}

fn compress(
    ids: &[usize],
    offset: usize,
    initial_vocab_size: usize,
    max_codebook_size: usize,
    max_subtokens: usize,
    max_out_seq_length: usize,
    eot_token_id: usize,
    disabled_ids: &Set,
) -> (Vec<usize>, Vec<Vec<usize>>, Option<usize>) {
    let mut compressed_ids: Vec<usize> = Vec::new();
    let mut codebook: FxHashMap<Vec<usize>, usize> = FxHashMap::default();

    let mut next_id: usize = initial_vocab_size;
    let mut ids_to_merge: Vec<usize> = Vec::with_capacity(max_subtokens);

    let mut i = 0;
    while i < ids.len() && i < max_out_seq_length {
        let id = ids[i + offset];

        if disabled_ids.contains(&id) {
            if ids_to_merge.len() > 0 {
                compressed_ids.push(get_usize_from_codebook(&codebook, &ids_to_merge));
                ids_to_merge.clear();
            }
            compressed_ids.push(id);
            i += 1;
            continue;
        }

        ids_to_merge.push(id);

        let is_in_codebook = codebook_contains(&codebook, &ids_to_merge, initial_vocab_size);
        if !is_in_codebook {
            if next_id < initial_vocab_size + max_codebook_size {
                codebook.insert(ids_to_merge.clone(), next_id);
                next_id += 1;
            }

            ids_to_merge.pop();
            compressed_ids.push(get_usize_from_codebook(&codebook, &ids_to_merge));
            ids_to_merge.clear();
            ids_to_merge.push(id);
        }

        if ids_to_merge.len() == max_subtokens {
            compressed_ids.push(get_usize_from_codebook(&codebook, &ids_to_merge));
            ids_to_merge.clear();
        }

        i += 1;
    }

    if ids_to_merge.len() > max_subtokens && compressed_ids.len() < max_out_seq_length {
        let last_id = ids_to_merge.pop().unwrap();
        compressed_ids.push(get_usize_from_codebook(&codebook, &ids_to_merge));
        ids_to_merge.clear();
        ids_to_merge.push(last_id);
    }

    if ids_to_merge.len() > 0 && compressed_ids.len() < max_out_seq_length {
        compressed_ids.push(get_usize_from_codebook(&codebook, &ids_to_merge));
    }

    let mut codebook_vec: Vec<Vec<usize>> = codebook
        .iter()
        .sorted_by(|(_, value), (_, value2)| value.cmp(value2))
        .map(|(key, _)| {
            let mut k = key.clone();
            k.resize(max_subtokens, eot_token_id);
            k
        })
        .collect::<Vec<Vec<usize>>>();

    codebook_vec.resize(max_codebook_size, vec![eot_token_id; max_subtokens]);

    let mut j = i;
    let mut has_remaining_docs = false;
    while j < ids.len() {
        if ids[j] == eot_token_id {
            has_remaining_docs = true;
            break;
        }
        j += 1;
    }

    let remaining_ids = if has_remaining_docs { Some(j) } else { None };

    (compressed_ids, codebook_vec, remaining_ids)
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
    let num_tokens = header[2];

    let disabled_ids = disabled_ids_to_set(Some(vec![args.eot_token_id]));

    let mut compressed_ids: Vec<usize> = Vec::new();
    let mut codebook_vec: Vec<usize> = Vec::new();

    let mut i: usize = 0;
    let mut pb = pbar(Some(num_tokens as usize));
    while i < num_tokens as usize {
        let (c_ids, c_codebook, remaining_ids) = compress(
            &ids,
            i,
            args.initial_vocab_size,
            args.max_codebook_size,
            args.max_subtokens,
            args.max_out_seq_length,
            args.eot_token_id,
            &disabled_ids,
        );
        let offset = remaining_ids.unwrap_or(num_tokens as usize - i);
        let _ = pb.update(min(offset, num_tokens as usize - i));
        i += offset;

        compressed_ids.extend(c_ids);
        codebook_vec.extend(c_codebook.iter().flatten().copied());
    }
    let _ = pb.close();

    println!("compressed_ids.len(): {}", compressed_ids.len());
    println!("codebook_vec.len(): {}", codebook_vec.len());

    header[2] = compressed_ids.len() as i32;
    header[3] = codebook_vec.len() as i32;
    header[4] = args.max_codebook_size as i32;
    header[5] = args.max_subtokens as i32;

    let mut compressed_file = File::create(format!("../{}/compressed_{}", args.name, filename)).unwrap();
    let header_bytes: Vec<u8> = header.iter().flat_map(|&x| x.to_le_bytes()).collect();
    compressed_file.write_all(&header_bytes).unwrap();
    let compressed_ids_bytes: Vec<u8> = compressed_ids.iter().flat_map(|&x| (x as u16).to_le_bytes()).collect();
    compressed_file.write_all(&compressed_ids_bytes).unwrap();

    let mut codebook_file = File::create(format!("../{}/codebooks_{}", args.name, filename)).unwrap();
    let codebook_bytes: Vec<u8> = codebook_vec.iter().flat_map(|&x| (x as u16).to_le_bytes()).collect();
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
