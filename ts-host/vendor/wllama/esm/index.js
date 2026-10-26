var __defProp = Object.defineProperty;
var __defProps = Object.defineProperties;
var __getOwnPropDescs = Object.getOwnPropertyDescriptors;
var __getOwnPropSymbols = Object.getOwnPropertySymbols;
var __hasOwnProp = Object.prototype.hasOwnProperty;
var __propIsEnum = Object.prototype.propertyIsEnumerable;
var __knownSymbol = (name, symbol) => (symbol = Symbol[name]) ? symbol : Symbol.for("Symbol." + name);
var __defNormalProp = (obj, key, value) => key in obj ? __defProp(obj, key, { enumerable: true, configurable: true, writable: true, value }) : obj[key] = value;
var __spreadValues = (a, b) => {
  for (var prop in b || (b = {}))
    if (__hasOwnProp.call(b, prop))
      __defNormalProp(a, prop, b[prop]);
  if (__getOwnPropSymbols)
    for (var prop of __getOwnPropSymbols(b)) {
      if (__propIsEnum.call(b, prop))
        __defNormalProp(a, prop, b[prop]);
    }
  return a;
};
var __spreadProps = (a, b) => __defProps(a, __getOwnPropDescs(b));
var __publicField = (obj, key, value) => __defNormalProp(obj, typeof key !== "symbol" ? key + "" : key, value);
var __async = (__this, __arguments, generator) => {
  return new Promise((resolve, reject) => {
    var fulfilled = (value) => {
      try {
        step(generator.next(value));
      } catch (e) {
        reject(e);
      }
    };
    var rejected = (value) => {
      try {
        step(generator.throw(value));
      } catch (e) {
        reject(e);
      }
    };
    var step = (x) => x.done ? resolve(x.value) : Promise.resolve(x.value).then(fulfilled, rejected);
    step((generator = generator.apply(__this, __arguments)).next());
  });
};
var __await = function(promise, isYieldStar) {
  this[0] = promise;
  this[1] = isYieldStar;
};
var __asyncGenerator = (__this, __arguments, generator) => {
  var resume = (k, v, yes, no) => {
    try {
      var x = generator[k](v), isAwait = (v = x.value) instanceof __await, done = x.done;
      Promise.resolve(isAwait ? v[0] : v).then((y) => isAwait ? resume(k === "return" ? k : "next", v[1] ? { done: y.done, value: y.value } : y, yes, no) : yes({ value: y, done })).catch((e) => resume("throw", e, yes, no));
    } catch (e) {
      no(e);
    }
  }, method = (k) => it[k] = (x) => new Promise((yes, no) => resume(k, x, yes, no)), it = {};
  return generator = generator.apply(__this, __arguments), it[__knownSymbol("asyncIterator")] = () => it, method("next"), method("throw"), method("return"), it;
};
var __forAwait = (obj, it, method) => (it = obj[__knownSymbol("asyncIterator")]) ? it.call(obj) : (obj = obj[__knownSymbol("iterator")](), it = {}, method = (key, fn) => (fn = obj[key]) && (it[key] = (arg) => new Promise((yes, no, done) => (arg = fn.call(obj, arg), done = arg.done, Promise.resolve(arg.value).then((value) => yes({ value, done }), no)))), method("next"), method("return"), it);

// src/glue/messages.ts
var GLUE_VERSION = 1;
var GLUE_MESSAGE_PROTOTYPES = {
  "erro_evt": {
    "name": "erro_evt",
    "structName": "glue_msg_error",
    "className": "GlueMsgError",
    "fields": [
      {
        "type": "str",
        "name": "message",
        "isNullable": false
      }
    ]
  },
  "load_req": {
    "name": "load_req",
    "structName": "glue_msg_load_req",
    "className": "GlueMsgLoadReq",
    "fields": [
      {
        "type": "arr_str",
        "name": "model_paths",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "mmproj_path",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "n_ctx_auto",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "use_mmap",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "use_mlock",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_gpu_layers",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_ctx",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_threads",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "model_alias",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "log_level",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "embeddings",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "offload_kqv",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "n_batch",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "n_ubatch",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "n_parallel",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "pooling_type",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "rope_scaling_type",
        "isNullable": true
      },
      {
        "type": "float",
        "name": "rope_freq_base",
        "isNullable": true
      },
      {
        "type": "float",
        "name": "rope_freq_scale",
        "isNullable": true
      },
      {
        "type": "float",
        "name": "yarn_ext_factor",
        "isNullable": true
      },
      {
        "type": "float",
        "name": "yarn_attn_factor",
        "isNullable": true
      },
      {
        "type": "float",
        "name": "yarn_beta_fast",
        "isNullable": true
      },
      {
        "type": "float",
        "name": "yarn_beta_slow",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "yarn_orig_ctx",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "cache_type_k",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "cache_type_v",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "kv_unified",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "flash_attn",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "swa_full",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "n_ctx_checkpoints",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "checkpoint_min_step",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "chat_template",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "jinja",
        "isNullable": true
      },
      {
        "type": "arr_str",
        "name": "default_template_kwargs_keys",
        "isNullable": true
      },
      {
        "type": "arr_str",
        "name": "default_template_kwargs_vals",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "reasoning",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "image_min_tokens",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "image_max_tokens",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "warmup",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "no_kv_offload",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "mmproj_offload",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "cont_batching",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "n_keep",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "ctx_shift",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "cache_idle_slots",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "n_cache_reuse",
        "isNullable": true
      },
      {
        "type": "arr_str",
        "name": "lora_paths",
        "isNullable": true
      },
      {
        "type": "arr_float",
        "name": "lora_scales",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "lora_init_without_apply",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "spec_draft_model",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "spec_draft_ngl",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "spec_draft_n_max",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "spec_draft_n_min",
        "isNullable": true
      },
      {
        "type": "float",
        "name": "spec_draft_p_min",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "spec_draft_threads",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "spec_draft_threads_batch",
        "isNullable": true
      },
      {
        "type": "arr_str",
        "name": "kv_overrides_keys",
        "isNullable": true
      },
      {
        "type": "arr_str",
        "name": "kv_overrides_vals",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "reasoning_budget_tokens",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "reasoning_budget_message",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "reasoning_format",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "skip_chat_parsing",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "prefill_assistant",
        "isNullable": true
      }
    ]
  },
  "load_res": {
    "name": "load_res",
    "structName": "glue_msg_load_res",
    "className": "GlueMsgLoadRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_ctx",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_batch",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_ubatch",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_vocab",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_ctx_train",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_embd",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_layer",
        "isNullable": false
      },
      {
        "type": "arr_str",
        "name": "metadata_key",
        "isNullable": false
      },
      {
        "type": "arr_str",
        "name": "metadata_val",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "token_bos",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "token_eos",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "token_eot",
        "isNullable": false
      },
      {
        "type": "arr_int",
        "name": "list_tokens_eog",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "add_bos_token",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "add_eos_token",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "has_encoder",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "token_decoder_start",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "media_marker",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "has_image_input",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "has_audio_input",
        "isNullable": false
      }
    ]
  },
  "cmpl_req": {
    "name": "cmpl_req",
    "structName": "glue_msg_completion_req",
    "className": "GlueMsgCompletionReq",
    "fields": [
      {
        "type": "bool",
        "name": "is_chat",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "data_json",
        "isNullable": false
      },
      {
        "type": "arr_raw",
        "name": "files",
        "isNullable": false
      }
    ]
  },
  "cmpl_res": {
    "name": "cmpl_res",
    "structName": "glue_msg_completion_res",
    "className": "GlueMsgCompletionRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "embd_req": {
    "name": "embd_req",
    "structName": "glue_msg_embedding_req",
    "className": "GlueMsgEmbeddingReq",
    "fields": [
      {
        "type": "str",
        "name": "data_json",
        "isNullable": false
      },
      {
        "type": "arr_raw",
        "name": "files",
        "isNullable": false
      }
    ]
  },
  "embd_res": {
    "name": "embd_res",
    "structName": "glue_msg_embedding_res",
    "className": "GlueMsgEmbeddingRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "rrnk_req": {
    "name": "rrnk_req",
    "structName": "glue_msg_rerank_req",
    "className": "GlueMsgRerankReq",
    "fields": [
      {
        "type": "str",
        "name": "data_json",
        "isNullable": false
      }
    ]
  },
  "rrnk_res": {
    "name": "rrnk_res",
    "structName": "glue_msg_rerank_res",
    "className": "GlueMsgRerankRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "sys1_req": {
    "name": "sys1_req",
    "structName": "glue_msg_systemone_req",
    "className": "GlueMsgSystemoneReq",
    "fields": [
      {
        "type": "str",
        "name": "data_json",
        "isNullable": false
      }
    ]
  },
  "sys1_res": {
    "name": "sys1_res",
    "structName": "glue_msg_systemone_res",
    "className": "GlueMsgSystemoneRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "gres_req": {
    "name": "gres_req",
    "structName": "glue_msg_get_result_req",
    "className": "GlueMsgGetResultReq",
    "fields": [
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "gres_res": {
    "name": "gres_res",
    "structName": "glue_msg_get_result_res",
    "className": "GlueMsgGetResultRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "has_more",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "is_error",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "data_json",
        "isNullable": false
      }
    ]
  },
  "cncl_req": {
    "name": "cncl_req",
    "structName": "glue_msg_cancel_req",
    "className": "GlueMsgCancelReq",
    "fields": [
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "cncl_res": {
    "name": "cncl_res",
    "structName": "glue_msg_cancel_res",
    "className": "GlueMsgCancelRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      }
    ]
  },
  "tbop_req": {
    "name": "tbop_req",
    "structName": "glue_msg_test_backend_ops_req",
    "className": "GlueMsgTestBackendOpsReq",
    "fields": [
      {
        "type": "arr_str",
        "name": "args",
        "isNullable": false
      }
    ]
  },
  "tbop_res": {
    "name": "tbop_res",
    "structName": "glue_msg_test_backend_ops_res",
    "className": "GlueMsgTestBackendOpsRes",
    "fields": [
      {
        "type": "int",
        "name": "retcode",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      }
    ]
  }
};

// src/glue/glue.ts
var GLUE_MAGIC = new Uint8Array([71, 76, 85, 69]);
var GLUE_DTYPE_NULL = 0;
var GLUE_DTYPE_BOOL = 1;
var GLUE_DTYPE_INT = 2;
var GLUE_DTYPE_FLOAT = 3;
var GLUE_DTYPE_STRING = 4;
var GLUE_DTYPE_RAW = 5;
var GLUE_DTYPE_ARRAY_BOOL = 6;
var GLUE_DTYPE_ARRAY_INT = 7;
var GLUE_DTYPE_ARRAY_FLOAT = 8;
var GLUE_DTYPE_ARRAY_STRING = 9;
var GLUE_DTYPE_ARRAY_RAW = 10;
var TYPE_MAP = {
  str: GLUE_DTYPE_STRING,
  int: GLUE_DTYPE_INT,
  float: GLUE_DTYPE_FLOAT,
  bool: GLUE_DTYPE_BOOL,
  raw: GLUE_DTYPE_RAW,
  arr_str: GLUE_DTYPE_ARRAY_STRING,
  arr_int: GLUE_DTYPE_ARRAY_INT,
  arr_float: GLUE_DTYPE_ARRAY_FLOAT,
  arr_bool: GLUE_DTYPE_ARRAY_BOOL,
  arr_raw: GLUE_DTYPE_ARRAY_RAW,
  null: GLUE_DTYPE_NULL
};
function glueDeserialize(buf) {
  let offset = 0;
  const view = new DataView(buf.buffer);
  const readUint32 = () => {
    const value = view.getUint32(offset, true);
    offset += 4;
    return value;
  };
  const readInt32 = () => {
    const value = view.getInt32(offset, true);
    offset += 4;
    return value;
  };
  const readFloat = () => {
    const value = view.getFloat32(offset, true);
    offset += 4;
    return value;
  };
  const readBool = () => {
    return readUint32() !== 0;
  };
  const readString = (customLen) => {
    const length = customLen != null ? customLen : readUint32();
    const value = new TextDecoder().decode(buf.slice(offset, offset + length));
    offset += length;
    return value;
  };
  const readRaw = () => {
    const length = readUint32();
    const value = buf.slice(offset, offset + length);
    offset += length;
    return value;
  };
  const readArray = (readItem) => {
    const length = readUint32();
    const value = new Array(length);
    for (let i = 0; i < length; i++) {
      value[i] = readItem();
    }
    return value;
  };
  const readNull = () => null;
  const readField = (field) => {
    switch (field.type) {
      case "str":
        return readString();
      case "int":
        return readInt32();
      case "float":
        return readFloat();
      case "bool":
        return readBool();
      case "raw":
        return readRaw();
      case "arr_str":
        return readArray(readString);
      case "arr_int":
        return readArray(readInt32);
      case "arr_float":
        return readArray(readFloat);
      case "arr_bool":
        return readArray(readBool);
      case "arr_raw":
        return readArray(readRaw);
      case "null":
        return readNull();
    }
  };
  const magicValid = buf[0] === GLUE_MAGIC[0] && buf[1] === GLUE_MAGIC[1] && buf[2] === GLUE_MAGIC[2] && buf[3] === GLUE_MAGIC[3];
  offset += 4;
  if (!magicValid) {
    throw new Error("Invalid magic number");
  }
  const version = readUint32();
  if (version !== GLUE_VERSION) {
    throw new Error("Invalid version number");
  }
  const name = readString(8);
  const msgProto = GLUE_MESSAGE_PROTOTYPES[name];
  if (!msgProto) {
    throw new Error(`Unknown message name: ${name}`);
  }
  const output = { _name: name };
  for (const field of msgProto.fields) {
    const readType = readUint32();
    if (readType === GLUE_DTYPE_NULL) {
      if (!field.isNullable) {
        throw new Error(
          `${name}: Expect field ${field.name} to be non-nullable`
        );
      }
      output[field.name] = null;
      continue;
    }
    if (readType !== TYPE_MAP[field.type]) {
      throw new Error(
        `${name}: Expect field ${field.name} to have type ${field.type}`
      );
    }
    output[field.name] = readField(field);
  }
  return output;
}
function glueSerialize(msg) {
  const msgProto = GLUE_MESSAGE_PROTOTYPES[msg._name];
  if (!msgProto) {
    throw new Error(`Unknown message name: ${msg._name}`);
  }
  const bufs = [];
  const writeUint32 = (value) => {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setUint32(0, value, true);
    bufs.push(new Uint8Array(buf));
  };
  const writeInt32 = (value) => {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setInt32(0, value, true);
    bufs.push(new Uint8Array(buf));
  };
  const writeFloat = (value) => {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setFloat32(0, value, true);
    bufs.push(new Uint8Array(buf));
  };
  const writeBool = (value) => {
    writeUint32(value ? 1 : 0);
  };
  const writeString = (value) => {
    const utf8 = new TextEncoder().encode(value);
    writeUint32(utf8.byteLength);
    bufs.push(utf8);
  };
  const writeRaw = (value) => {
    writeUint32(value.byteLength);
    bufs.push(value);
  };
  const writeArray = (value, writeItem) => {
    writeUint32(value.length);
    for (const item of value) {
      writeItem(item);
    }
  };
  const writeNull = () => {
  };
  bufs.push(GLUE_MAGIC);
  writeUint32(GLUE_VERSION);
  {
    const utf8 = new TextEncoder().encode(msg._name);
    bufs.push(utf8);
  }
  for (const field of msgProto.fields) {
    const val = msg[field.name];
    if (!field.isNullable && (val === null || val === void 0)) {
      throw new Error(
        `${msg._name}: Expect field ${field.name} to be non-nullable`
      );
    }
    if (val === null || val === void 0) {
      writeUint32(GLUE_DTYPE_NULL);
      continue;
    }
    writeUint32(TYPE_MAP[field.type]);
    switch (field.type) {
      case "str":
        writeString(val);
        break;
      case "int":
        writeInt32(val);
        break;
      case "float":
        writeFloat(val);
        break;
      case "bool":
        writeBool(val);
        break;
      case "raw":
        writeRaw(val);
        break;
      case "arr_str":
        writeArray(val, writeString);
        break;
      case "arr_int":
        writeArray(val, writeInt32);
        break;
      case "arr_float":
        writeArray(val, writeFloat);
        break;
      case "arr_bool":
        writeArray(val, writeBool);
        break;
      case "arr_raw":
        writeArray(val, writeRaw);
        break;
      case "null":
        writeNull();
        break;
    }
  }
  const totalLength = bufs.reduce((acc, buf) => acc + buf.byteLength, 0);
  const output = new Uint8Array(totalLength);
  let offset = 0;
  for (const buf of bufs) {
    output.set(buf, offset);
    offset += buf.byteLength;
  }
  return output;
}

// src/wasm/source-map.ts
var WASM_SOURCE_MAP = {
  "default": "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",
  "compat": "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"
};

// src/debug.ts
var cache = /* @__PURE__ */ new Map();
function loadMap(buildKey) {
  return __async(this, null, function* () {
    if (cache.has(buildKey)) return cache.get(buildKey);
    const b64 = WASM_SOURCE_MAP[buildKey];
    if (!b64) throw new Error(`No source map for build "${buildKey}"`);
    const gzipped = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
    const ds = new DecompressionStream("gzip");
    const writer = ds.writable.getWriter();
    writer.write(gzipped);
    writer.close();
    const buf = yield new Response(ds.readable).arrayBuffer();
    const dv = new DataView(buf);
    const bytes = new Uint8Array(buf);
    const firstId = dv.getUint32(0, true);
    const funcCount = dv.getUint32(4, true);
    const numNames = dv.getUint32(8, true);
    const td = new TextDecoder();
    const names = [];
    let pos = 12;
    for (let i = 0; i < numNames; i++) {
      const len = bytes[pos++];
      names.push(td.decode(bytes.subarray(pos, pos + len)));
      pos += len;
    }
    const funcNames = [];
    for (let i = 0; i < funcCount; i++) {
      const idx = dv.getUint16(pos, true);
      pos += 2;
      funcNames.push(idx === 65535 ? null : names[idx]);
    }
    const entry = { firstId, funcNames };
    cache.set(buildKey, entry);
    return entry;
  });
}
var Debug = {
  /**
   * Resolves a list of wasm function indices to their cleaned symbol names.
   */
  decodeFuncIds: (funcIds, isCompatBuild) => __async(void 0, null, function* () {
    const buildKey = isCompatBuild ? "compat" : "default";
    const { firstId, funcNames } = yield loadMap(buildKey);
    return funcIds.map((funcId) => {
      const i = funcId - firstId;
      const name = i >= 0 && i < funcNames.length && funcNames[i] ? funcNames[i] : "(unknown)";
      return { funcId, name };
    });
  }),
  /**
   * Annotates a wasm stack trace string with resolved function names.
   *
   * Example input from Chrome:
   *   at http://localhost:8080/esm/wasm/wllama.wasm:wasm-function[775]:0x74251
   *   at async blob:http://localhost:8080/53a863cc-7227-45cc-8594-ddbbf5257f20:317:28
   *
   * Example input from Firefox:
   *   @http://localhost:8080/esm/wasm/wllama.wasm:wasm-function[796]:0x7dfe2
   *       at wModuleInit/WebAssembly.promising/< (9b6a2acd-d909-44e2-b021-d42fb9087cfb:15:32) index.js:1433:45
   *
   * Example input from Safari:
   *   2441@wasm-function[2441]
   *       at wrapper (d746f19e-4523-4f36-ba06-d0969acc0b05:22:126009)
   *
   * Example output:
   *   wasm-func[775] (server_response::send)
   */
  decodeStackTrace: (stack, isCompatBuild) => __async(void 0, null, function* () {
    const re = /wasm-function\[(\d+)\]/g;
    const funcIds = [
      ...new Set([...stack.matchAll(re)].map((m) => parseInt(m[1])))
    ];
    if (funcIds.length === 0) return stack;
    const resolved = yield Debug.decodeFuncIds(funcIds, isCompatBuild);
    return resolved.map((r) => {
      if (r.name === "(unknown)") {
        return `    wasm-func[${r.funcId}] (unknown)`;
      }
      return `    wasm-func[${r.funcId}] (${r.name})`;
    }).join("\n");
  })
};

// src/utils.ts
var textDecoder = new TextDecoder();
var URL_PARTS_REGEX = /-(\d{5})-of-(\d{5})\.gguf(?:\?.*)?$/;
var parseShardNumber = (fnameOrUrl) => {
  const matches = fnameOrUrl.match(URL_PARTS_REGEX);
  if (!matches) {
    return {
      baseURL: fnameOrUrl,
      current: 1,
      total: 1
    };
  } else {
    return {
      baseURL: fnameOrUrl.replace(URL_PARTS_REGEX, ""),
      current: parseInt(matches[1]),
      total: parseInt(matches[2])
    };
  }
};
var sortFileByShard = (blobs) => {
  const isFiles = blobs.every((b) => !!b.name);
  if (isFiles && blobs.length > 1) {
    const files = blobs;
    files.sort((a, b) => {
      const infoA = parseShardNumber(a.name);
      const infoB = parseShardNumber(b.name);
      return infoA.current - infoB.current;
    });
  }
};
var isMmproj = (blob) => __async(void 0, null, function* () {
  const META_NAME = "general.architecture";
  const META_VAL = "clip";
  const tmp = blob.slice(0, 128 * 1024);
  const header = yield tmp.arrayBuffer();
  const buf = new Uint8Array(header);
  const nameBytes = new TextEncoder().encode(META_NAME);
  const valBytes = new TextEncoder().encode(META_VAL);
  let offset = -1;
  outer: for (let i = 0; i <= buf.length - nameBytes.length; i++) {
    for (let j = 0; j < nameBytes.length; j++) {
      if (buf[i + j] !== nameBytes[j]) continue outer;
    }
    offset = i;
    break;
  }
  if (offset === -1) return false;
  if (offset + 8 * 4 + 4 > buf.length) return false;
  const view = new DataView(header);
  const valLen = view.getBigUint64(offset + 8 * 3, true);
  if (valLen !== /* @__PURE__ */ BigInt("4")) return false;
  for (let i = 0; i < valBytes.length; i++) {
    if (buf[offset + 8 * 4 + i] !== valBytes[i]) return false;
  }
  return true;
});
var absoluteUrl = (relativePath) => typeof document === "undefined" ? new URL(relativePath, self.location.href).href : new URL(relativePath, document.baseURI).href;
var padDigits = (number, digits) => {
  return Array(Math.max(digits - String(number).length + 1, 0)).join("0") + number;
};
var sumArr = (arr) => arr.reduce((prev, curr) => prev + curr, 0);
var isString = (value) => !!(value == null ? void 0 : value.startsWith);
var MMPROJ_FILE_NAME = "mmproj.gguf";
var prepareBlobs = (blobsInp) => __async(void 0, null, function* () {
  const blobs = [];
  let blobMmproj = null;
  for (const blob of blobsInp) {
    if (yield isMmproj(blob)) {
      blobMmproj = blob;
    } else {
      blobs.push(blob);
    }
  }
  sortFileByShard(blobs);
  const result = blobs.map((blob, i) => ({
    blob,
    name: `model-${padDigits(i + 1, 5)}-of-${padDigits(blobs.length, 5)}.gguf`
  }));
  if (blobMmproj) {
    result.push({
      blob: blobMmproj,
      name: MMPROJ_FILE_NAME
    });
  }
  return {
    llm: result.filter((f) => f.name !== MMPROJ_FILE_NAME),
    mmproj: blobMmproj ? { blob: blobMmproj, name: MMPROJ_FILE_NAME } : null,
    all: result
  };
});
var isSupportMultiThread = () => ((e) => __async(void 0, null, function* () {
  try {
    return "undefined" != typeof MessageChannel && new MessageChannel().port1.postMessage(new SharedArrayBuffer(1)), WebAssembly.validate(e);
  } catch (e2) {
    return false;
  }
}))(
  new Uint8Array([
    0,
    97,
    115,
    109,
    1,
    0,
    0,
    0,
    1,
    4,
    1,
    96,
    0,
    0,
    3,
    2,
    1,
    0,
    5,
    4,
    1,
    3,
    1,
    1,
    10,
    11,
    1,
    9,
    0,
    65,
    0,
    254,
    16,
    2,
    0,
    26,
    11
  ])
);
var isSupportExceptions = () => __async(void 0, null, function* () {
  return WebAssembly.validate(
    new Uint8Array([
      0,
      97,
      115,
      109,
      1,
      0,
      0,
      0,
      1,
      4,
      1,
      96,
      0,
      0,
      3,
      2,
      1,
      0,
      10,
      8,
      1,
      6,
      0,
      6,
      64,
      25,
      11,
      11
    ])
  );
});
var isSupportSIMD = () => __async(void 0, null, function* () {
  return WebAssembly.validate(
    new Uint8Array([
      0,
      97,
      115,
      109,
      1,
      0,
      0,
      0,
      1,
      5,
      1,
      96,
      0,
      1,
      123,
      3,
      2,
      1,
      0,
      10,
      10,
      1,
      8,
      0,
      65,
      0,
      253,
      15,
      253,
      98,
      11
    ])
  );
});
var isSupportJSPI = () => {
  return !!WebAssembly.Suspending;
};
var isSupportWebGPU = () => {
  return !!navigator.gpu;
};
var isSupportMem64 = () => {
  try {
    new WebAssembly.Memory({
      address: "i64",
      initial: /* @__PURE__ */ BigInt("1")
      // 1 page (64 KiB)
    });
    return true;
  } catch (e) {
    return false;
  }
};
var checkEnvironmentCompatible = () => __async(void 0, null, function* () {
  if (!(yield isSupportExceptions())) {
    throw new Error("WebAssembly runtime does not support exception handling");
  }
  if (!(yield isSupportSIMD())) {
    throw new Error("WebAssembly runtime does not support SIMD");
  }
});
var isFirefox = () => {
  return !!navigator.userAgent.match(/Firefox\/([0-9\.]+)(?:\s|$)/);
};
var GGUF_FILE_REGEX = /^.*\.gguf(?:\?.*)?$/;
var isValidGgufFile = (path) => {
  return GGUF_FILE_REGEX.test(path);
};
var isSafariMobile = () => {
  return !!navigator.userAgent.match(/Version\/([0-9\._]+).*Mobile.*Safari.*/);
};
var createWorker = (workerCode) => {
  const workerURL = URL.createObjectURL(
    isString(workerCode) ? new Blob([workerCode], { type: "text/javascript" }) : workerCode
  );
  return new Worker(workerURL, { type: "module" });
};
var cbToAsyncIter = (fn) => (...args) => {
  let values = [];
  let resolve;
  let reject;
  values.push(
    new Promise((res, rej) => {
      resolve = res;
      reject = rej;
    })
  );
  fn(...args, (val, done, err) => {
    if (err) {
      reject(err);
      return;
    }
    resolve([val, done]);
    values.push(
      new Promise((res, rej) => {
        resolve = res;
        reject = rej;
      })
    );
  });
  return function() {
    return __asyncGenerator(this, null, function* () {
      let val;
      for (let i = 0, done = false; !done; i++) {
        [val, done] = yield new __await(values[i]);
        delete values[i];
        if (val !== void 0) yield val;
      }
    });
  }();
};
var canUseAsyncFileRead = (compat) => isSupportJSPI() || compat;
var needCompat = () => !isSupportJSPI() || !isSupportMem64();

// src/workers-code/generated.ts
var LIBLLAMA_VERSION = "b11720-ee85d96";
var LLAMA_CPP_WORKER_CODE = "// Start the main llama.cpp\nlet wllamaMalloc;\nlet wllamaStart;\nlet wllamaAction;\nlet wllamaExit;\nlet wllamaDebug;\n\nlet Module = null;\nlet isCompat = false;\nlet lastStack = '';\nlet isAborted = false;\nlet hasMultithread = false;\n\n//////////////////////////////////////////////////////////////\n// UTILS\n//////////////////////////////////////////////////////////////\n\n// send message back to main thread\nconst msg = (data, transfer) => postMessage(data, transfer);\n\n// Convert CPP log into JS log\nconst cppLogToJSLog = (line) => {\n  const matched = line.match(/@@(DEBUG|INFO|WARN|ERROR)@@(.*)/);\n  return !!matched\n    ? {\n        level: (matched[1] === 'INFO' ? 'debug' : matched[1]).toLowerCase(),\n        text: matched[2],\n      }\n    : { level: 'log', text: line };\n};\n\nconst getHeapU8 = () => {\n  const buffer = Module.wasmMemory.buffer;\n  return new Uint8Array(buffer);\n};\n\nconst toSizeT = (num) => {\n  return isCompat ? Number(num) : BigInt(num);\n};\n\n// Get module config that forwards stdout/err to main thread\nconst getWModuleConfig = (_argMainScriptBlob) => {\n  var pathConfig = RUN_OPTIONS.pathConfig;\n  var pthreadPoolSize = RUN_OPTIONS.nbThread;\n  var argMainScriptBlob = _argMainScriptBlob;\n\n  isCompat = RUN_OPTIONS.compat;\n  hasMultithread = pthreadPoolSize > 1;\n\n  msg({\n    verb: 'console.debug',\n    args: [\n      `Multithread enabled: ${hasMultithread}, pthreadPoolSize: ${pthreadPoolSize}`,\n    ],\n  });\n\n  if (!pathConfig['wllama.wasm']) {\n    throw new Error('\"wllama.wasm\" is missing in pathConfig');\n  }\n  return {\n    noInitialRun: true,\n    print: function (text) {\n      if (arguments.length > 1)\n        text = Array.prototype.slice.call(arguments).join(' ');\n      msg({ verb: 'console.log', args: [text] });\n    },\n    printErr: function (text) {\n      if (arguments.length > 1)\n        text = Array.prototype.slice.call(arguments).join(' ');\n      if (text.startsWith('@@STACK@@')) {\n        lastStack = text.slice('@@STACK@@'.length);\n        return;\n      }\n      const logLine = cppLogToJSLog(text);\n      msg({ verb: 'console.' + logLine.level, args: [logLine.text] });\n    },\n    locateFile: function (filename, basePath) {\n      const p = pathConfig[filename];\n      const truncate = (str) =>\n        str.length > 128 ? `${str.substr(0, 128)}...` : str;\n      if (filename.match(/wllama\\.worker\\.js/)) {\n        msg({\n          verb: 'console.error',\n          args: [\n            '\"wllama.worker.js\" is removed from v2.2.1. Hint: make sure to clear browser\\'s cache.',\n          ],\n        });\n      } else {\n        msg({\n          verb: 'console.debug',\n          args: [`Loading \"${filename}\" from \"${truncate(p)}\"`],\n        });\n        return p;\n      }\n    },\n    mainScriptUrlOrBlob: hasMultithread\n      ? argMainScriptBlob\n      : 'throw new Error(\"Multithreading is not enabled\")',\n    pthreadPoolSize: hasMultithread ? pthreadPoolSize : 0,\n    wasmMemory: hasMultithread ? getWasmMemory() : null,\n    onAbort: function (message) {\n      isAborted = true;\n      msg({ verb: 'signal.abort', args: ['abort', message, lastStack, null] });\n    },\n    onExit: function (code) {\n      isAborted = true;\n      const callstack = new Error().stack.toString();\n      msg({\n        verb: 'signal.abort',\n        args: ['abort', 'exit(' + code + ')', callstack, null],\n      });\n    },\n  };\n};\n\n// Get the memory to be used by wasm. (Only used in multi-thread mode)\n// Because we have a weird OOM issue on iOS, we need to try some values\n// See: https://github.com/emscripten-core/emscripten/issues/19144\n//      https://github.com/godotengine/godot/issues/70621\nconst getWasmMemory = () => {\n  let minBytes = 128 * 1024 * 1024;\n  let maxBytes = 4096 * 1024 * 1024;\n  let stepBytes = 128 * 1024 * 1024;\n  while (maxBytes > minBytes) {\n    try {\n      const wasmMemory = new WebAssembly.Memory({\n        initial: toSizeT(minBytes / 65536),\n        maximum: toSizeT(maxBytes / 65536),\n        shared: true,\n        address: isCompat ? undefined : 'i64',\n      });\n      return wasmMemory;\n    } catch (e) {\n      maxBytes -= stepBytes;\n      continue; // retry\n    }\n  }\n  throw new Error('Cannot allocate WebAssembly.Memory');\n};\n\n//////////////////////////////////////////////////////////////\n// HEAPFS PATCH\n//////////////////////////////////////////////////////////////\n\n/**\n * By default, emscripten uses memfs. The way it works is by\n * allocating new Uint8Array in javascript heap. This is not good\n * because it requires files to be copied to wasm heap each time\n * a file is read.\n *\n * HeapFS is an alternative, which resolves this problem by\n * allocating space for file directly inside wasm heap. This\n * allows us to mmap without doing any copy.\n *\n * For llama.cpp, this is great because we use MAP_SHARED\n *\n * Ref: https://github.com/ngxson/wllama/pull/39\n * Ref: https://github.com/emscripten-core/emscripten/blob/main/src/library_memfs.js\n *\n * Note 29/05/2024 @ngxson\n * Due to ftell() being limited to MAX_LONG, we cannot load files bigger than 2^31 bytes (or 2GB)\n * Ref: https://github.com/emscripten-core/emscripten/blob/main/system/lib/libc/musl/src/stdio/ftell.c\n */\n\nconst fsNameToFile = {}; // map Name => File\nconst fsIdToFile = {}; // map ID => File\nlet currFileId = 0;\n\n// Patch and redirect memfs calls to wllama\nconst patchHeapFS = () => {\n  const m = Module;\n  // save functions\n  m.MEMFS.stream_ops._read = m.MEMFS.stream_ops.read;\n  m.MEMFS.stream_ops._write = m.MEMFS.stream_ops.write;\n  m.MEMFS.stream_ops._llseek = m.MEMFS.stream_ops.llseek;\n  m.MEMFS.stream_ops._allocate = m.MEMFS.stream_ops.allocate;\n  m.MEMFS.stream_ops._mmap = m.MEMFS.stream_ops.mmap;\n  m.MEMFS.stream_ops._msync = m.MEMFS.stream_ops.msync;\n\n  const patchStream = (stream) => {\n    const name = stream.node.name;\n    if (fsNameToFile[name]) {\n      const f = fsNameToFile[name];\n      const ptr = Number(f.ptr);\n      stream.node.contents = getHeapU8().subarray(ptr, ptr + f.size);\n      stream.node.usedBytes = f.size;\n    }\n  };\n\n  // replace \"read\" functions\n  m.MEMFS.stream_ops.read = function (\n    stream,\n    buffer,\n    offset,\n    length,\n    position\n  ) {\n    patchStream(stream);\n    return m.MEMFS.stream_ops._read(stream, buffer, offset, length, position);\n  };\n  m.MEMFS.ops_table.file.stream.read = m.MEMFS.stream_ops.read;\n\n  // replace \"llseek\" functions\n  m.MEMFS.stream_ops.llseek = function (stream, offset, whence) {\n    patchStream(stream);\n    return m.MEMFS.stream_ops._llseek(stream, offset, whence);\n  };\n  m.MEMFS.ops_table.file.stream.llseek = m.MEMFS.stream_ops.llseek;\n\n  // replace \"mmap\" functions\n  m.MEMFS.stream_ops.mmap = function (stream, length, position, prot, flags) {\n    patchStream(stream);\n    const name = stream.node.name;\n    if (fsNameToFile[name]) {\n      const f = fsNameToFile[name];\n      const mmapPtr = f.ptr + toSizeT(position);\n      return {\n        ptr: mmapPtr,\n        allocated: false,\n      };\n    } else {\n      return m.MEMFS.stream_ops._mmap(stream, length, position, prot, flags);\n    }\n  };\n  m.MEMFS.ops_table.file.stream.mmap = m.MEMFS.stream_ops.mmap;\n\n  // mount FS\n  m.FS.mkdir('/models');\n  m.FS.mount(m.MEMFS, { root: '.' }, '/models');\n};\n\n// Allocate a new file in wllama heapfs, returns file ID\nconst heapfsAlloc = (name, size, allocBuffer) => {\n  if (size < 1) {\n    throw new Error('File size must be bigger than 0');\n  }\n  const m = Module;\n  const ptr = toSizeT(allocBuffer ? m.mmapAlloc(size) : 0);\n  const file = {\n    ptr: ptr,\n    size: size,\n    id: currFileId++,\n  };\n  fsIdToFile[file.id] = file;\n  fsNameToFile[name] = file;\n  return file.id;\n};\n\n// Add new file to wllama heapfs, return number of written bytes\nconst heapfsWrite = (id, buffer, offset) => {\n  if (fsIdToFile[id]) {\n    const { ptr, size } = fsIdToFile[id];\n    const afterWriteByte = offset + buffer.byteLength;\n    if (afterWriteByte > size) {\n      throw new Error(\n        `File ID ${id} write out of bound, afterWriteByte = ${afterWriteByte} while size = ${size}`\n      );\n    }\n    getHeapU8().set(buffer, Number(ptr) + offset);\n    return buffer.byteLength;\n  } else {\n    throw new Error(`File ID ${id} not found in heapfs`);\n  }\n};\n\n//////////////////////////////////////////////////////////////\n// ASYNC FILE READ\n//////////////////////////////////////////////////////////////\n\nlet isAwaitReading = false;\nlet pendingReadPromise = null;\nlet pendingReadResolve = null;\nlet pendingReadReject = null;\n\nconst _stripModelsPrefix = (path) => path.replace(/^\\/?models\\//, '');\n\n// Called from EM_ASYNC_JS stub in wllama-fs.h (path is already a JS string)\nconst _wllama_js_file_read = async (path, offset, req_size, out_ptr) => {\n  const name = _stripModelsPrefix(path);\n\n  pendingReadPromise = new Promise((res, rej) => {\n    pendingReadResolve = res;\n    pendingReadReject = rej;\n  });\n  isAwaitReading = true;\n\n  postMessage({ verb: 'fs.read_req', args: [name, offset, req_size] });\n\n  let data;\n  try {\n    data = await pendingReadPromise;\n  } finally {\n    isAwaitReading = false;\n    pendingReadResolve = null;\n    pendingReadReject = null;\n  }\n\n  const bytes = new Uint8Array(data);\n  getHeapU8().set(bytes, out_ptr);\n  return toSizeT(bytes.length);\n};\n\n//////////////////////////////////////////////////////////////\n// MAIN CODE\n//////////////////////////////////////////////////////////////\n\nconst callWrapper = (name, ret, args, isAsync) => {\n  const fn = Module.cwrap(\n    name,\n    ret,\n    args,\n    isAsync ? { async: true } : undefined\n  );\n  return async (action, req) => {\n    // console.log(`Calling ${name} with action:`, action, 'and req:', req);\n    let result;\n    try {\n      if (args.length === 2) {\n        result = isAsync ? await fn(action, req) : fn(action, req);\n      } else {\n        result = fn();\n      }\n    } catch (ex) {\n      console.error(ex);\n      throw ex;\n    }\n    return result;\n  };\n};\n\n// re-entering the wasm while a call is suspended (JSPI / asyncify) corrupts its state, so only one call runs at a time and the rest wait in the queue\nlet wasmCallBusy = false;\nconst wasmCallQueue = [];\n\nconst runWasmCall = async (callbackId, fn) => {\n  if (isAborted) {\n    // the wasm is dead, fail fast instead of calling into it\n    msg({ callbackId, err: 'wllama has crashed, please reload the module' });\n    return;\n  }\n  if (wasmCallBusy) {\n    wasmCallQueue.push({ callbackId, fn });\n    return;\n  }\n  wasmCallBusy = true;\n  try {\n    await fn();\n  } finally {\n    wasmCallBusy = false;\n    if (isAborted) {\n      // do not touch the wasm again after it aborted; the main thread already rejected the queued tasks\n      wasmCallQueue.length = 0;\n    } else {\n      const next = wasmCallQueue.shift();\n      if (next) runWasmCall(next.callbackId, next.fn);\n    }\n  }\n};\n\nconst runAction = async (data) => {\n  const { args, callbackId } = data;\n  const argAction = args[0];\n  const argEncodedMsg = args[1];\n  try {\n    const inputPtr = await wllamaMalloc(toSizeT(argEncodedMsg.byteLength), 0);\n    // copy data to wasm heap\n    const inputBuffer = new Uint8Array(\n      getHeapU8().buffer,\n      Number(inputPtr),\n      argEncodedMsg.byteLength\n    );\n    inputBuffer.set(argEncodedMsg, 0);\n    const outputPtr = await wllamaAction(argAction, inputPtr);\n    // length of output buffer is written at the first 4 bytes of input buffer\n    const outputLen = new Uint32Array(\n      getHeapU8().buffer,\n      Number(inputPtr),\n      1\n    )[0];\n    // copy the output buffer to JS heap\n    const outputBuffer = new Uint8Array(outputLen);\n    const outputSrcView = new Uint8Array(\n      getHeapU8().buffer,\n      Number(outputPtr),\n      outputLen\n    );\n    outputBuffer.set(outputSrcView, 0); // copy it\n    msg({ callbackId, result: outputBuffer }, [outputBuffer.buffer]);\n  } catch (err) {\n    handleError(err);\n  }\n};\n\nfunction handleError(err) {\n  // If WASM already aborted, onAbort already sent signal.abort; skip to avoid\n  // re-reporting the resulting WebAssembly.RuntimeError as a JS exception.\n  if (isAborted) return;\n\n  const message = err ? err.message || String(err) : 'Unknown error';\n  const stack = err ? err.stack || String(err) : '';\n  msg({\n    verb: 'signal.abort',\n    args: ['exception', message, stack, err],\n  });\n}\n\nonmessage = async (e) => {\n  if (!e.data) return;\n  const { verb, args, callbackId } = e.data;\n\n  // fs.read_res arrives while wasm is JSPI-suspended; resolve the pending promise.\n  if (verb === 'fs.read_res') {\n    if (pendingReadResolve) {\n      pendingReadResolve(args[0]);\n    }\n    return;\n  }\n\n  // Guard: while awaiting a file read, reject any other incoming task.\n  if (isAwaitReading) {\n    if (callbackId) {\n      msg({\n        callbackId,\n        err: 'Worker is suspended waiting for file data (JSPI)',\n      });\n    }\n    return;\n  }\n\n  if (!callbackId) {\n    msg({ verb: 'console.error', args: ['callbackId is required', e.data] });\n    return;\n  }\n\n  if (verb === 'module.init') {\n    const argMainScriptBlob = args[0];\n    const argUseAsyncFile = args[1];\n    try {\n      Module = getWModuleConfig(argMainScriptBlob);\n      Module.preRun = () => {\n        if (argUseAsyncFile) {\n          Module.ENV['USE_ASYNC_FILE'] = '1';\n        }\n      };\n      Module.onRuntimeInitialized = () => {\n        // async call once module is ready\n        // init FS\n        patchHeapFS();\n        // init cwrap\n        const pointer = isCompat ? 'number' : 'bigint';\n        // TODO: note sure why emscripten cannot bind if there is only 1 argument\n        wllamaMalloc = callWrapper('wllama_malloc', pointer, [\n          'number',\n          pointer,\n        ]);\n        wllamaStart = callWrapper('wllama_start', 'string', [], true);\n        wllamaAction = callWrapper(\n          'wllama_action',\n          pointer,\n          ['string', pointer],\n          true\n        );\n        wllamaExit = callWrapper('wllama_exit', 'string', []);\n        wllamaDebug = callWrapper('wllama_debug', 'string', []);\n        msg({ callbackId, result: null });\n      };\n      wModuleInit();\n    } catch (err) {\n      handleError(err);\n    }\n    return;\n  }\n\n  if (verb === 'fs.alloc') {\n    const argFilename = args[0];\n    const argSize = args[1];\n    const argAllocBuffer = args[2];\n    try {\n      // create blank file\n      const emptyBuffer = new ArrayBuffer(0);\n      Module['FS_createDataFile'](\n        '/models',\n        argFilename,\n        emptyBuffer,\n        true,\n        true,\n        true\n      );\n      // alloc data on heap\n      const fileId = heapfsAlloc(argFilename, argSize, argAllocBuffer);\n      msg({ callbackId, result: { fileId } });\n    } catch (err) {\n      handleError(err);\n    }\n    return;\n  }\n\n  if (verb === 'fs.write') {\n    const argFileId = args[0];\n    const argBuffer = args[1];\n    const argOffset = args[2];\n    try {\n      const writtenBytes = heapfsWrite(argFileId, argBuffer, argOffset);\n      msg({ callbackId, result: { writtenBytes } });\n    } catch (err) {\n      handleError(err);\n    }\n    return;\n  }\n\n  if (verb === 'wllama.start') {\n    await runWasmCall(callbackId, async () => {\n      try {\n        const result = await wllamaStart();\n        msg({ callbackId, result });\n      } catch (err) {\n        handleError(err);\n      }\n    });\n    return;\n  }\n\n  if (verb === 'wllama.action') {\n    await runWasmCall(callbackId, () => runAction(e.data));\n    return;\n  }\n\n  if (verb === 'wllama.exit') {\n    await runWasmCall(callbackId, async () => {\n      try {\n        const result = await wllamaExit();\n        msg({ callbackId, result });\n      } catch (err) {\n        handleError(err);\n      }\n    });\n    return;\n  }\n\n  if (verb === 'wllama.debug') {\n    await runWasmCall(callbackId, async () => {\n      try {\n        const result = await wllamaDebug();\n        msg({ callbackId, result });\n      } catch (err) {\n        handleError(err);\n      }\n    });\n    return;\n  }\n};\n";
var OPFS_UTILS_WORKER_CODE = "let accessHandle;\nlet abortController = new AbortController();\n\nasync function openFile(filename) {\n  const opfsRoot = await navigator.storage.getDirectory();\n  const cacheDir = await opfsRoot.getDirectoryHandle('cache', { create: true });\n  const fileHandler = await cacheDir.getFileHandle(filename, { create: true });\n  accessHandle = await fileHandler.createSyncAccessHandle();\n  accessHandle.truncate(0); // clear file content\n}\n\nasync function writeFile(buf) {\n  accessHandle.write(buf);\n}\n\nasync function closeFile() {\n  accessHandle.flush();\n  accessHandle.close();\n}\n\nasync function writeTextFile(filename, str) {\n  await openFile(filename);\n  await writeFile(new TextEncoder().encode(str));\n  await closeFile();\n}\n\nconst throttled = (func, delay) => {\n  let lastRun = 0;\n  return (...args) => {\n    const now = Date.now();\n    if (now - lastRun > delay) {\n      lastRun = now;\n      func.apply(null, args);\n    }\n  };\n};\n\nconst assertNonNull = (val) => {\n  if (val === null || val === undefined) {\n    throw new Error('OPFS Worker: Assertion failed');\n  }\n};\n\n// respond to main thread\nconst resOK = () => postMessage({ ok: true });\nconst resProgress = (loaded, total) =>\n  postMessage({ progress: { loaded, total } });\nconst resErr = (err) => postMessage({ err });\n\nonmessage = async (e) => {\n  try {\n    if (!e.data) return;\n\n    /**\n     * @param {Object} e.data\n     *\n     * Fine-control FS actions:\n     * - { action: 'open', filename: 'string' }\n     * - { action: 'write', buf: ArrayBuffer }\n     * - { action: 'close' }\n     *\n     * Simple write API:\n     * - { action: 'write-simple', filename: 'string', buf: ArrayBuffer }\n     *\n     * Download API:\n     * - { action: 'download', url: 'string', filename: 'string', options: Object, metadataFileName: 'string' }\n     * - { action: 'download-abort' }\n     */\n    const {\n      action,\n      filename,\n      buf,\n      url,\n      options,\n      metadataFileName,\n      metadataAdditional,\n    } = e.data;\n\n    if (action === 'open') {\n      assertNonNull(filename);\n      await openFile(filename);\n      return resOK();\n    } else if (action === 'write') {\n      assertNonNull(buf);\n      await writeFile(buf);\n      return resOK();\n    } else if (action === 'close') {\n      await closeFile();\n      return resOK();\n    } else if (action === 'write-simple') {\n      assertNonNull(filename);\n      assertNonNull(buf);\n      await openFile(filename);\n      await writeFile(buf);\n      await closeFile();\n      return resOK();\n    } else if (action === 'download') {\n      assertNonNull(url);\n      assertNonNull(filename);\n      assertNonNull(metadataFileName);\n      assertNonNull(options);\n      assertNonNull(options.aborted);\n      abortController = new AbortController();\n      if (options.aborted) abortController.abort();\n      const response = await fetch(url, {\n        ...options,\n        signal: abortController.signal,\n      });\n      const contentLength = response.headers.get('content-length');\n      const etag = (response.headers.get('etag') || '').replace(\n        /[^A-Za-z0-9]/g,\n        ''\n      );\n      const total = parseInt(contentLength, 10);\n      const reader = response.body.getReader();\n      await openFile(filename);\n      let loaded = 0;\n      const throttledProgress = throttled(resProgress, 100);\n      while (true) {\n        const { done, value } = await reader.read();\n        if (done) break;\n        loaded += value.byteLength;\n        await writeFile(value);\n        throttledProgress(loaded, total);\n      }\n      resProgress(total, total); // 100% done\n      await closeFile();\n      // make sure this is in-sync with CacheEntryMetadata\n      await writeTextFile(\n        metadataFileName,\n        JSON.stringify({\n          originalURL: url,\n          originalSize: total,\n          etag,\n          ...metadataAdditional,\n        })\n      );\n      return resOK();\n    } else if (action === 'download-abort') {\n      if (abortController) {\n        abortController.abort();\n      }\n      return;\n    }\n\n    throw new Error('OPFS Worker: Invalid action', e.data);\n  } catch (err) {\n    return resErr(err);\n  }\n};\n";
var WLLAMA_EMSCRIPTEN_CODE = 'var Module=typeof Module!="undefined"?Module:{};var ENVIRONMENT_IS_WEB=!!globalThis.window;var ENVIRONMENT_IS_WORKER=!!globalThis.WorkerGlobalScope;var ENVIRONMENT_IS_NODE=globalThis.process?.versions?.node&&globalThis.process?.type!="renderer";var ENVIRONMENT_IS_PTHREAD=ENVIRONMENT_IS_WORKER&&self.name?.startsWith("em-pthread");if(ENVIRONMENT_IS_NODE){var worker_threads=require("worker_threads");global.Worker=worker_threads.Worker;ENVIRONMENT_IS_WORKER=!worker_threads.isMainThread;ENVIRONMENT_IS_PTHREAD=ENVIRONMENT_IS_WORKER&&worker_threads["workerData"]=="em-pthread"}var arguments_=[];var thisProgram="./this.program";var quit_=(status,toThrow)=>{throw toThrow};var _scriptName=globalThis.document?.currentScript?.src;if(typeof __filename!="undefined"){_scriptName=__filename}else if(ENVIRONMENT_IS_WORKER){_scriptName=self.location.href}var scriptDirectory="";function locateFile(path){if(Module["locateFile"]){return Module["locateFile"](path,scriptDirectory)}return scriptDirectory+path}var readAsync,readBinary;if(ENVIRONMENT_IS_NODE){var fs=require("fs");scriptDirectory=__dirname+"/";readBinary=filename=>{filename=isFileURI(filename)?new URL(filename):filename;var ret=fs.readFileSync(filename);return ret};readAsync=async(filename,binary=true)=>{filename=isFileURI(filename)?new URL(filename):filename;var ret=fs.readFileSync(filename,binary?undefined:"utf8");return ret};if(process.argv.length>1){thisProgram=process.argv[1].replace(/\\\\/g,"/")}arguments_=process.argv.slice(2);if(typeof module!="undefined"){module["exports"]=Module}quit_=(status,toThrow)=>{process.exitCode=status;throw toThrow}}else if(ENVIRONMENT_IS_WEB||ENVIRONMENT_IS_WORKER){try{scriptDirectory=new URL(".",_scriptName).href}catch{}if(!ENVIRONMENT_IS_NODE){if(ENVIRONMENT_IS_WORKER){readBinary=url=>{var xhr=new XMLHttpRequest;xhr.open("GET",url,false);xhr.responseType="arraybuffer";xhr.send(null);return new Uint8Array(xhr.response)}}readAsync=async url=>{if(isFileURI(url)){return new Promise((resolve,reject)=>{var xhr=new XMLHttpRequest;xhr.open("GET",url,true);xhr.responseType="arraybuffer";xhr.onload=()=>{if(xhr.status==200||xhr.status==0&&xhr.response){resolve(xhr.response);return}reject(xhr.status)};xhr.onerror=reject;xhr.send(null)})}var response=await fetch(url,{credentials:"same-origin"});if(response.ok){return response.arrayBuffer()}throw new Error(response.status+" : "+response.url)}}}else{}var defaultPrint=console.log.bind(console);var defaultPrintErr=console.error.bind(console);if(ENVIRONMENT_IS_NODE){var utils=require("util");var stringify=a=>typeof a=="object"?utils.inspect(a):a;defaultPrint=(...args)=>fs.writeSync(1,args.map(stringify).join(" ")+"\\n");defaultPrintErr=(...args)=>fs.writeSync(2,args.map(stringify).join(" ")+"\\n")}var out=defaultPrint;var err=defaultPrintErr;var wasmBinary;var wasmModule;var ABORT=false;var EXITSTATUS;function assert(condition,text){if(!condition){abort(text)}}var isFileURI=filename=>filename.startsWith("file://");function growMemViews(){if(wasmMemory.buffer!=HEAP8.buffer){updateMemoryViews()}}if(ENVIRONMENT_IS_NODE&&ENVIRONMENT_IS_PTHREAD){var parentPort=worker_threads["parentPort"];parentPort.on("message",msg=>global.onmessage?.({data:msg}));Object.assign(globalThis,{self:global,postMessage:msg=>parentPort["postMessage"](msg)});process.on("uncaughtException",err=>{postMessage({cmd:"uncaughtException",error:err});process.exit(1)})}var startWorker;if(ENVIRONMENT_IS_PTHREAD){var initializedJS=false;self.onunhandledrejection=e=>{throw e.reason||e};async function handleMessage(e){try{var msgData=e["data"];var cmd=msgData.cmd;if(cmd==="load"){let messageQueue=[];self.onmessage=e=>messageQueue.push(e);startWorker=()=>{postMessage({cmd:"loaded"});for(let msg of messageQueue){handleMessage(msg)}self.onmessage=handleMessage};for(const handler of msgData.handlers){if(!Module[handler]||Module[handler].proxy){Module[handler]=(...args)=>{postMessage({cmd:"callHandler",handler,args})};if(handler=="print")out=Module[handler];if(handler=="printErr")err=Module[handler]}}wasmMemory=msgData.wasmMemory;updateMemoryViews();wasmModule=msgData.wasmModule;createWasm();run()}else if(cmd==="run"){establishStackSpace(msgData.pthread_ptr);__emscripten_thread_init(msgData.pthread_ptr,0,0,1,0,0);PThread.threadInitTLS();__emscripten_thread_mailbox_await(msgData.pthread_ptr);if(!initializedJS){initializedJS=true}try{await invokeEntryPoint(msgData.start_routine,msgData.arg)}catch(ex){if(ex!="unwind"){throw ex}}}else if(msgData.target==="setimmediate"){}else if(cmd==="checkMailbox"){if(initializedJS){checkMailbox()}}else if(cmd){err(`worker: received unknown command ${cmd}`);err(msgData)}}catch(ex){__emscripten_thread_crashed();throw ex}}self.onmessage=handleMessage}var HEAP8,HEAPU8,HEAP16,HEAPU16,HEAP32,HEAPU32,HEAPF32,HEAPF64;var HEAP64,HEAPU64;var runtimeInitialized=false;function updateMemoryViews(){var b=wasmMemory.buffer;HEAP8=new Int8Array(b);HEAP16=new Int16Array(b);Module["HEAPU8"]=HEAPU8=new Uint8Array(b);HEAPU16=new Uint16Array(b);HEAP32=new Int32Array(b);HEAPU32=new Uint32Array(b);HEAPF32=new Float32Array(b);HEAPF64=new Float64Array(b);HEAP64=new BigInt64Array(b);HEAPU64=new BigUint64Array(b)}function initMemory(){if(ENVIRONMENT_IS_PTHREAD){return}if(Module["wasmMemory"]){wasmMemory=Module["wasmMemory"]}else{var INITIAL_MEMORY=Module["INITIAL_MEMORY"]||134217728;wasmMemory=new WebAssembly.Memory({initial:BigInt(INITIAL_MEMORY/65536),maximum:65536n,shared:true,address:"i64"})}updateMemoryViews()}function preRun(){if(Module["preRun"]){if(typeof Module["preRun"]=="function")Module["preRun"]=[Module["preRun"]];while(Module["preRun"].length){addOnPreRun(Module["preRun"].shift())}}callRuntimeCallbacks(onPreRuns)}function initRuntime(){runtimeInitialized=true;if(ENVIRONMENT_IS_PTHREAD)return startWorker();if(!Module["noFSInit"]&&!FS.initialized)FS.init();TTY.init();wasmExports["__wasm_call_ctors"]();FS.ignorePermissions=false}function preMain(){}function postRun(){if(ENVIRONMENT_IS_PTHREAD){return}if(Module["postRun"]){if(typeof Module["postRun"]=="function")Module["postRun"]=[Module["postRun"]];while(Module["postRun"].length){addOnPostRun(Module["postRun"].shift())}}callRuntimeCallbacks(onPostRuns)}function abort(what){Module["onAbort"]?.(what);what="Aborted("+what+")";err(what);ABORT=true;what+=". Build with -sASSERTIONS for more info.";if(runtimeInitialized){___trap()}var e=new WebAssembly.RuntimeError(what);throw e}var wasmBinaryFile;function findWasmBinary(){return locateFile("wllama.wasm")}function getBinarySync(file){if(file==wasmBinaryFile&&wasmBinary){return new Uint8Array(wasmBinary)}if(readBinary){return readBinary(file)}throw"both async and sync fetching of the wasm failed"}async function getWasmBinary(binaryFile){if(!wasmBinary){try{var response=await readAsync(binaryFile);return new Uint8Array(response)}catch{}}return getBinarySync(binaryFile)}async function instantiateArrayBuffer(binaryFile,imports){try{var binary=await getWasmBinary(binaryFile);var instance=await WebAssembly.instantiate(binary,imports);return instance}catch(reason){err(`failed to asynchronously prepare wasm: ${reason}`);abort(reason)}}async function instantiateAsync(binary,binaryFile,imports){if(!binary&&!isFileURI(binaryFile)&&!ENVIRONMENT_IS_NODE){try{var response=fetch(binaryFile,{credentials:"same-origin"});var instantiationResult=await WebAssembly.instantiateStreaming(response,imports);return instantiationResult}catch(reason){err(`wasm streaming compile failed: ${reason}`);err("falling back to ArrayBuffer instantiation")}}return instantiateArrayBuffer(binaryFile,imports)}function getWasmImports(){assignWasmImports();if(!wasmImports.__instrumented){wasmImports.__instrumented=true;Asyncify.instrumentWasmImports(wasmImports)}var imports={env:wasmImports,wasi_snapshot_preview1:wasmImports};return imports}async function createWasm(){function receiveInstance(instance,module){wasmExports=instance.exports;wasmExports=Asyncify.instrumentWasmExports(wasmExports);wasmExports=applySignatureConversions(wasmExports);registerTLSInit(wasmExports["_emscripten_tls_init"]);assignWasmExports(wasmExports);wasmModule=module;removeRunDependency("wasm-instantiate");return wasmExports}addRunDependency("wasm-instantiate");function receiveInstantiationResult(result){return receiveInstance(result["instance"],result["module"])}var info=getWasmImports();if(Module["instantiateWasm"]){return new Promise((resolve,reject)=>{Module["instantiateWasm"](info,(inst,mod)=>{resolve(receiveInstance(inst,mod))})})}if(ENVIRONMENT_IS_PTHREAD){var instance=new WebAssembly.Instance(wasmModule,getWasmImports());return receiveInstance(instance,wasmModule)}wasmBinaryFile??=findWasmBinary();var result=await instantiateAsync(wasmBinary,wasmBinaryFile,info);var exports=receiveInstantiationResult(result);return exports}class ExitStatus{name="ExitStatus";constructor(status){this.message=`Program terminated with exit(${status})`;this.status=status}}var terminateWorker=worker=>{worker.terminate();worker.onmessage=e=>{}};var cleanupThread=pthread_ptr=>{var worker=PThread.pthreads[pthread_ptr];PThread.returnWorkerToPool(worker)};var callRuntimeCallbacks=callbacks=>{while(callbacks.length>0){callbacks.shift()(Module)}};var onPreRuns=[];var addOnPreRun=cb=>onPreRuns.push(cb);var runDependencies=0;var dependenciesFulfilled=null;var removeRunDependency=id=>{runDependencies--;Module["monitorRunDependencies"]?.(runDependencies);if(runDependencies==0){if(dependenciesFulfilled){var callback=dependenciesFulfilled;dependenciesFulfilled=null;callback()}}};var addRunDependency=id=>{runDependencies++;Module["monitorRunDependencies"]?.(runDependencies)};var spawnThread=threadParams=>{var worker=PThread.getNewWorker();if(!worker){return 6}PThread.runningWorkers.push(worker);PThread.pthreads[threadParams.pthread_ptr]=worker;worker.pthread_ptr=threadParams.pthread_ptr;var msg={cmd:"run",start_routine:threadParams.startRoutine,arg:threadParams.arg,pthread_ptr:threadParams.pthread_ptr};if(ENVIRONMENT_IS_NODE){worker.unref()}worker.postMessage(msg,threadParams.transferList);return 0};var runtimeKeepaliveCounter=0;var keepRuntimeAlive=()=>noExitRuntime||runtimeKeepaliveCounter>0;var stackSave=()=>_emscripten_stack_get_current();var stackRestore=val=>__emscripten_stack_restore(val);var stackAlloc=sz=>__emscripten_stack_alloc(sz);var proxyToMainThread=(funcIndex,emAsmAddr,sync,...callArgs)=>{var serializedNumCallArgs=callArgs.length*2;var sp=stackSave();var args=stackAlloc(serializedNumCallArgs*8);var b=args/8;for(var i=0;i<callArgs.length;i++){var arg=callArgs[i];if(typeof arg=="bigint"){(growMemViews(),HEAP64)[b+2*i]=1n;(growMemViews(),HEAP64)[b+2*i+1]=arg}else{(growMemViews(),HEAP64)[b+2*i]=0n;(growMemViews(),HEAPF64)[b+2*i+1]=arg}}var rtn=__emscripten_run_js_on_main_thread(funcIndex,emAsmAddr,serializedNumCallArgs,args,sync);stackRestore(sp);return rtn};function _proc_exit(code){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(0,0,1,code);EXITSTATUS=code;if(!keepRuntimeAlive()){PThread.terminateAllThreads();Module["onExit"]?.(code);ABORT=true}quit_(code,new ExitStatus(code))}function exitOnMainThread(returnCode){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(1,0,0,returnCode);_exit(returnCode)}var exitJS=(status,implicit)=>{EXITSTATUS=status;if(ENVIRONMENT_IS_PTHREAD){exitOnMainThread(status);throw"unwind"}_proc_exit(status)};var _exit=exitJS;var PThread={unusedWorkers:[],runningWorkers:[],tlsInitFunctions:[],pthreads:{},init(){if(!ENVIRONMENT_IS_PTHREAD){PThread.initMainThread()}},initMainThread(){var pthreadPoolSize=Module["pthreadPoolSize"];while(pthreadPoolSize--){PThread.allocateUnusedWorker()}addOnPreRun(async()=>{var pthreadPoolReady=PThread.loadWasmModuleToAllWorkers();addRunDependency("loading-workers");await pthreadPoolReady;removeRunDependency("loading-workers")})},terminateAllThreads:()=>{for(var worker of PThread.runningWorkers){terminateWorker(worker)}for(var worker of PThread.unusedWorkers){terminateWorker(worker)}PThread.unusedWorkers=[];PThread.runningWorkers=[];PThread.pthreads={}},returnWorkerToPool:worker=>{var pthread_ptr=worker.pthread_ptr;delete PThread.pthreads[pthread_ptr];PThread.unusedWorkers.push(worker);PThread.runningWorkers.splice(PThread.runningWorkers.indexOf(worker),1);worker.pthread_ptr=0;__emscripten_thread_free_data(pthread_ptr)},threadInitTLS(){PThread.tlsInitFunctions.forEach(f=>f())},loadWasmModuleToWorker:worker=>new Promise(onFinishedLoading=>{worker.onmessage=e=>{var d=e["data"];var cmd=d.cmd;if(d.targetThread&&d.targetThread!=_pthread_self()){var targetWorker=PThread.pthreads[d.targetThread];if(targetWorker){targetWorker.postMessage(d,d.transferList)}else{err(`Internal error! Worker sent a message "${cmd}" to target pthread ${d.targetThread}, but that thread no longer exists!`)}return}if(cmd==="checkMailbox"){checkMailbox()}else if(cmd==="spawnThread"){spawnThread(d)}else if(cmd==="cleanupThread"){callUserCallback(()=>cleanupThread(d.thread))}else if(cmd==="loaded"){worker.loaded=true;if(ENVIRONMENT_IS_NODE&&!worker.pthread_ptr){worker.unref()}onFinishedLoading(worker)}else if(d.target==="setimmediate"){worker.postMessage(d)}else if(cmd==="uncaughtException"){worker.onerror(d.error)}else if(cmd==="callHandler"){Module[d.handler](...d.args)}else if(cmd){err(`worker sent an unknown command ${cmd}`)}};worker.onerror=e=>{var message="worker sent an error!";err(`${message} ${e.filename}:${e.lineno}: ${e.message}`);throw e};if(ENVIRONMENT_IS_NODE){worker.on("message",data=>worker.onmessage({data}));worker.on("error",e=>worker.onerror(e))}var handlers=[];var knownHandlers=["onExit","onAbort","print","printErr"];for(var handler of knownHandlers){if(Module.propertyIsEnumerable(handler)){handlers.push(handler)}}worker.postMessage({cmd:"load",handlers,wasmMemory,wasmModule})}),async loadWasmModuleToAllWorkers(){if(ENVIRONMENT_IS_PTHREAD){return}let pthreadPoolReady=Promise.all(PThread.unusedWorkers.map(PThread.loadWasmModuleToWorker));return pthreadPoolReady},allocateUnusedWorker(){var worker;var pthreadMainJs=_scriptName;if(Module["mainScriptUrlOrBlob"]){pthreadMainJs=Module["mainScriptUrlOrBlob"];if(typeof pthreadMainJs!="string"){pthreadMainJs=URL.createObjectURL(pthreadMainJs)}}worker=new Worker(pthreadMainJs,{workerData:"em-pthread",name:"em-pthread"});PThread.unusedWorkers.push(worker)},getNewWorker(){if(PThread.unusedWorkers.length==0){PThread.allocateUnusedWorker();PThread.loadWasmModuleToWorker(PThread.unusedWorkers[0])}return PThread.unusedWorkers.pop()}};var onPostRuns=[];var addOnPostRun=cb=>onPostRuns.push(cb);function establishStackSpace(pthread_ptr){var stackHigh=Number((growMemViews(),HEAPU64)[(pthread_ptr+88)/8]);var stackSize=Number((growMemViews(),HEAPU64)[(pthread_ptr+96)/8]);var stackLow=stackHigh-stackSize;_emscripten_stack_set_limits(stackHigh,stackLow);stackRestore(stackHigh)}var wasmTableMirror=[];var getWasmTableEntry=funcPtr=>{funcPtr=Number(funcPtr);var func=wasmTableMirror[funcPtr];if(!func){wasmTableMirror[funcPtr]=func=wasmTable.get(BigInt(funcPtr));if(Asyncify.isAsyncExport(func)){wasmTableMirror[funcPtr]=func=Asyncify.makeAsyncFunction(func)}}return func};var invokeEntryPoint=async(ptr,arg)=>{runtimeKeepaliveCounter=0;noExitRuntime=0;var result=(a1=>WebAssembly.promising(getWasmTableEntry(ptr)).call(null,BigInt(a1)))(arg);function finish(result){if(keepRuntimeAlive()){EXITSTATUS=result;return}__emscripten_thread_exit(result)}result=await result;finish(result)};invokeEntryPoint.isAsync=true;var noExitRuntime=true;var registerTLSInit=tlsInitFunc=>PThread.tlsInitFunctions.push(tlsInitFunc);var wasmMemory;function pthreadCreateProxied(pthread_ptr,attr,startRoutine,arg){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(2,0,1,pthread_ptr,attr,startRoutine,arg);return ___pthread_create_js(pthread_ptr,attr,startRoutine,arg)}var _emscripten_has_threading_support=()=>!!globalThis.SharedArrayBuffer;var INT53_MAX=9007199254740992;var INT53_MIN=-9007199254740992;var bigintToI53Checked=num=>num<INT53_MIN||num>INT53_MAX?NaN:Number(num);function ___pthread_create_js(pthread_ptr,attr,startRoutine,arg){pthread_ptr=bigintToI53Checked(pthread_ptr);attr=bigintToI53Checked(attr);startRoutine=bigintToI53Checked(startRoutine);arg=bigintToI53Checked(arg);if(!_emscripten_has_threading_support()){return 6}var transferList=[];var error=0;if(ENVIRONMENT_IS_PTHREAD&&(transferList.length===0||error)){return pthreadCreateProxied(pthread_ptr,attr,startRoutine,arg)}if(error)return error;var threadParams={startRoutine,pthread_ptr,arg,transferList};if(ENVIRONMENT_IS_PTHREAD){threadParams.cmd="spawnThread";postMessage(threadParams,transferList);return 0}return spawnThread(threadParams)}var syscallGetVarargP=()=>{var ret=Number((growMemViews(),HEAPU64)[SYSCALLS.varargs/8]);SYSCALLS.varargs+=8;return ret};var syscallGetVarargI=()=>{var ret=(growMemViews(),HEAP32)[+SYSCALLS.varargs/4];SYSCALLS.varargs+=4;return ret};var PATH={isAbs:path=>path.charAt(0)==="/",splitPath:filename=>{var splitPathRe=/^(\\/?|)([\\s\\S]*?)((?:\\.{1,2}|[^\\/]+?|)(\\.[^.\\/]*|))(?:[\\/]*)$/;return splitPathRe.exec(filename).slice(1)},normalizeArray:(parts,allowAboveRoot)=>{var up=0;for(var i=parts.length-1;i>=0;i--){var last=parts[i];if(last==="."){parts.splice(i,1)}else if(last===".."){parts.splice(i,1);up++}else if(up){parts.splice(i,1);up--}}if(allowAboveRoot){for(;up;up--){parts.unshift("..")}}return parts},normalize:path=>{var isAbsolute=PATH.isAbs(path),trailingSlash=path.slice(-1)==="/";path=PATH.normalizeArray(path.split("/").filter(p=>!!p),!isAbsolute).join("/");if(!path&&!isAbsolute){path="."}if(path&&trailingSlash){path+="/"}return(isAbsolute?"/":"")+path},dirname:path=>{var result=PATH.splitPath(path),root=result[0],dir=result[1];if(!root&&!dir){return"."}if(dir){dir=dir.slice(0,-1)}return root+dir},basename:path=>path&&path.match(/([^\\/]+|\\/)\\/*$/)[1],join:(...paths)=>PATH.normalize(paths.join("/")),join2:(l,r)=>PATH.normalize(l+"/"+r)};var initRandomFill=()=>view=>view.set(crypto.getRandomValues(new Uint8Array(view.byteLength)));var randomFill=view=>{(randomFill=initRandomFill())(view)};var PATH_FS={resolve:(...args)=>{var resolvedPath="",resolvedAbsolute=false;for(var i=args.length-1;i>=-1&&!resolvedAbsolute;i--){var path=i>=0?args[i]:FS.cwd();if(typeof path!="string"){throw new TypeError("Arguments to path.resolve must be strings")}else if(!path){return""}resolvedPath=path+"/"+resolvedPath;resolvedAbsolute=PATH.isAbs(path)}resolvedPath=PATH.normalizeArray(resolvedPath.split("/").filter(p=>!!p),!resolvedAbsolute).join("/");return(resolvedAbsolute?"/":"")+resolvedPath||"."},relative:(from,to)=>{from=PATH_FS.resolve(from).slice(1);to=PATH_FS.resolve(to).slice(1);function trim(arr){var start=0;for(;start<arr.length;start++){if(arr[start]!=="")break}var end=arr.length-1;for(;end>=0;end--){if(arr[end]!=="")break}if(start>end)return[];return arr.slice(start,end-start+1)}var fromParts=trim(from.split("/"));var toParts=trim(to.split("/"));var length=Math.min(fromParts.length,toParts.length);var samePartsLength=length;for(var i=0;i<length;i++){if(fromParts[i]!==toParts[i]){samePartsLength=i;break}}var outputParts=[];for(var i=samePartsLength;i<fromParts.length;i++){outputParts.push("..")}outputParts=outputParts.concat(toParts.slice(samePartsLength));return outputParts.join("/")}};var UTF8Decoder=globalThis.TextDecoder&&new TextDecoder;var findStringEnd=(heapOrArray,idx,maxBytesToRead,ignoreNul)=>{var maxIdx=idx+maxBytesToRead;if(ignoreNul)return maxIdx;while(heapOrArray[idx]&&!(idx>=maxIdx))++idx;return idx};var UTF8ArrayToString=(heapOrArray,idx=0,maxBytesToRead,ignoreNul)=>{var endPtr=findStringEnd(heapOrArray,idx,maxBytesToRead,ignoreNul);if(endPtr-idx>16&&heapOrArray.buffer&&UTF8Decoder){return UTF8Decoder.decode(heapOrArray.buffer instanceof ArrayBuffer?heapOrArray.subarray(idx,endPtr):heapOrArray.slice(idx,endPtr))}var str="";while(idx<endPtr){var u0=heapOrArray[idx++];if(!(u0&128)){str+=String.fromCharCode(u0);continue}var u1=heapOrArray[idx++]&63;if((u0&224)==192){str+=String.fromCharCode((u0&31)<<6|u1);continue}var u2=heapOrArray[idx++]&63;if((u0&240)==224){u0=(u0&15)<<12|u1<<6|u2}else{u0=(u0&7)<<18|u1<<12|u2<<6|heapOrArray[idx++]&63}if(u0<65536){str+=String.fromCharCode(u0)}else{var ch=u0-65536;str+=String.fromCharCode(55296|ch>>10,56320|ch&1023)}}return str};var FS_stdin_getChar_buffer=[];var lengthBytesUTF8=str=>{var len=0;for(var i=0;i<str.length;++i){var c=str.charCodeAt(i);if(c<=127){len++}else if(c<=2047){len+=2}else if(c>=55296&&c<=57343){len+=4;++i}else{len+=3}}return len};var stringToUTF8Array=(str,heap,outIdx,maxBytesToWrite)=>{if(!(maxBytesToWrite>0))return 0;var startIdx=outIdx;var endIdx=outIdx+maxBytesToWrite-1;for(var i=0;i<str.length;++i){var u=str.codePointAt(i);if(u<=127){if(outIdx>=endIdx)break;heap[outIdx++]=u}else if(u<=2047){if(outIdx+1>=endIdx)break;heap[outIdx++]=192|u>>6;heap[outIdx++]=128|u&63}else if(u<=65535){if(outIdx+2>=endIdx)break;heap[outIdx++]=224|u>>12;heap[outIdx++]=128|u>>6&63;heap[outIdx++]=128|u&63}else{if(outIdx+3>=endIdx)break;heap[outIdx++]=240|u>>18;heap[outIdx++]=128|u>>12&63;heap[outIdx++]=128|u>>6&63;heap[outIdx++]=128|u&63;i++}}heap[outIdx]=0;return outIdx-startIdx};var intArrayFromString=(stringy,dontAddNull,length)=>{var len=length>0?length:lengthBytesUTF8(stringy)+1;var u8array=new Array(len);var numBytesWritten=stringToUTF8Array(stringy,u8array,0,u8array.length);if(dontAddNull)u8array.length=numBytesWritten;return u8array};var FS_stdin_getChar=()=>{if(!FS_stdin_getChar_buffer.length){var result=null;if(ENVIRONMENT_IS_NODE){var BUFSIZE=256;var buf=Buffer.alloc(BUFSIZE);var bytesRead=0;var fd=process.stdin.fd;try{bytesRead=fs.readSync(fd,buf,0,BUFSIZE)}catch(e){if(e.toString().includes("EOF"))bytesRead=0;else throw e}if(bytesRead>0){result=buf.slice(0,bytesRead).toString("utf-8")}}else if(globalThis.window?.prompt){result=window.prompt("Input: ");if(result!==null){result+="\\n"}}else{}if(!result){return null}FS_stdin_getChar_buffer=intArrayFromString(result,true)}return FS_stdin_getChar_buffer.shift()};var TTY={ttys:[],init(){},shutdown(){},register(dev,ops){TTY.ttys[dev]={input:[],output:[],ops};FS.registerDevice(dev,TTY.stream_ops)},stream_ops:{open(stream){var tty=TTY.ttys[stream.node.rdev];if(!tty){throw new FS.ErrnoError(43)}stream.tty=tty;stream.seekable=false},close(stream){stream.tty.ops.fsync(stream.tty)},fsync(stream){stream.tty.ops.fsync(stream.tty)},read(stream,buffer,offset,length,pos){if(!stream.tty||!stream.tty.ops.get_char){throw new FS.ErrnoError(60)}var bytesRead=0;for(var i=0;i<length;i++){var result;try{result=stream.tty.ops.get_char(stream.tty)}catch(e){throw new FS.ErrnoError(29)}if(result===undefined&&bytesRead===0){throw new FS.ErrnoError(6)}if(result===null||result===undefined)break;bytesRead++;buffer[offset+i]=result}if(bytesRead){stream.node.atime=Date.now()}return bytesRead},write(stream,buffer,offset,length,pos){if(!stream.tty||!stream.tty.ops.put_char){throw new FS.ErrnoError(60)}try{for(var i=0;i<length;i++){stream.tty.ops.put_char(stream.tty,buffer[offset+i])}}catch(e){throw new FS.ErrnoError(29)}if(length){stream.node.mtime=stream.node.ctime=Date.now()}return i}},default_tty_ops:{get_char(tty){return FS_stdin_getChar()},put_char(tty,val){if(val===null||val===10){out(UTF8ArrayToString(tty.output));tty.output=[]}else{if(val!=0)tty.output.push(val)}},fsync(tty){if(tty.output?.length>0){out(UTF8ArrayToString(tty.output));tty.output=[]}},ioctl_tcgets(tty){return{c_iflag:25856,c_oflag:5,c_cflag:191,c_lflag:35387,c_cc:[3,28,127,21,4,0,1,0,17,19,26,0,18,15,23,22,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}},ioctl_tcsets(tty,optional_actions,data){return 0},ioctl_tiocgwinsz(tty){return[24,80]}},default_tty1_ops:{put_char(tty,val){if(val===null||val===10){err(UTF8ArrayToString(tty.output));tty.output=[]}else{if(val!=0)tty.output.push(val)}},fsync(tty){if(tty.output?.length>0){err(UTF8ArrayToString(tty.output));tty.output=[]}}}};var zeroMemory=(ptr,size)=>(growMemViews(),HEAPU8).fill(0,ptr,ptr+size);var alignMemory=(size,alignment)=>Math.ceil(size/alignment)*alignment;var mmapAlloc=size=>{size=alignMemory(size,65536);var ptr=_emscripten_builtin_memalign(65536,size);if(ptr)zeroMemory(ptr,size);return ptr};var MEMFS={ops_table:null,mount(mount){return MEMFS.createNode(null,"/",16895,0)},createNode(parent,name,mode,dev){if(FS.isBlkdev(mode)||FS.isFIFO(mode)){throw new FS.ErrnoError(63)}MEMFS.ops_table||={dir:{node:{getattr:MEMFS.node_ops.getattr,setattr:MEMFS.node_ops.setattr,lookup:MEMFS.node_ops.lookup,mknod:MEMFS.node_ops.mknod,rename:MEMFS.node_ops.rename,unlink:MEMFS.node_ops.unlink,rmdir:MEMFS.node_ops.rmdir,readdir:MEMFS.node_ops.readdir,symlink:MEMFS.node_ops.symlink},stream:{llseek:MEMFS.stream_ops.llseek}},file:{node:{getattr:MEMFS.node_ops.getattr,setattr:MEMFS.node_ops.setattr},stream:{llseek:MEMFS.stream_ops.llseek,read:MEMFS.stream_ops.read,write:MEMFS.stream_ops.write,mmap:MEMFS.stream_ops.mmap,msync:MEMFS.stream_ops.msync}},link:{node:{getattr:MEMFS.node_ops.getattr,setattr:MEMFS.node_ops.setattr,readlink:MEMFS.node_ops.readlink},stream:{}},chrdev:{node:{getattr:MEMFS.node_ops.getattr,setattr:MEMFS.node_ops.setattr},stream:FS.chrdev_stream_ops}};var node=FS.createNode(parent,name,mode,dev);if(FS.isDir(node.mode)){node.node_ops=MEMFS.ops_table.dir.node;node.stream_ops=MEMFS.ops_table.dir.stream;node.contents={}}else if(FS.isFile(node.mode)){node.node_ops=MEMFS.ops_table.file.node;node.stream_ops=MEMFS.ops_table.file.stream;node.usedBytes=0;node.contents=null}else if(FS.isLink(node.mode)){node.node_ops=MEMFS.ops_table.link.node;node.stream_ops=MEMFS.ops_table.link.stream}else if(FS.isChrdev(node.mode)){node.node_ops=MEMFS.ops_table.chrdev.node;node.stream_ops=MEMFS.ops_table.chrdev.stream}node.atime=node.mtime=node.ctime=Date.now();if(parent){parent.contents[name]=node;parent.atime=parent.mtime=parent.ctime=node.atime}return node},getFileDataAsTypedArray(node){if(!node.contents)return new Uint8Array(0);if(node.contents.subarray)return node.contents.subarray(0,node.usedBytes);return new Uint8Array(node.contents)},expandFileStorage(node,newCapacity){var prevCapacity=node.contents?node.contents.length:0;if(prevCapacity>=newCapacity)return;var CAPACITY_DOUBLING_MAX=1024*1024;newCapacity=Math.max(newCapacity,prevCapacity*(prevCapacity<CAPACITY_DOUBLING_MAX?2:1.125)>>>0);if(prevCapacity!=0)newCapacity=Math.max(newCapacity,256);var oldContents=node.contents;node.contents=new Uint8Array(newCapacity);if(node.usedBytes>0)node.contents.set(oldContents.subarray(0,node.usedBytes),0)},resizeFileStorage(node,newSize){if(node.usedBytes==newSize)return;if(newSize==0){node.contents=null;node.usedBytes=0}else{var oldContents=node.contents;node.contents=new Uint8Array(newSize);if(oldContents){node.contents.set(oldContents.subarray(0,Math.min(newSize,node.usedBytes)))}node.usedBytes=newSize}},node_ops:{getattr(node){var attr={};attr.dev=FS.isChrdev(node.mode)?node.id:1;attr.ino=node.id;attr.mode=node.mode;attr.nlink=1;attr.uid=0;attr.gid=0;attr.rdev=node.rdev;if(FS.isDir(node.mode)){attr.size=4096}else if(FS.isFile(node.mode)){attr.size=node.usedBytes}else if(FS.isLink(node.mode)){attr.size=node.link.length}else{attr.size=0}attr.atime=new Date(node.atime);attr.mtime=new Date(node.mtime);attr.ctime=new Date(node.ctime);attr.blksize=4096;attr.blocks=Math.ceil(attr.size/attr.blksize);return attr},setattr(node,attr){for(const key of["mode","atime","mtime","ctime"]){if(attr[key]!=null){node[key]=attr[key]}}if(attr.size!==undefined){MEMFS.resizeFileStorage(node,attr.size)}},lookup(parent,name){if(!MEMFS.doesNotExistError){MEMFS.doesNotExistError=new FS.ErrnoError(44);MEMFS.doesNotExistError.stack="<generic error, no stack>"}throw MEMFS.doesNotExistError},mknod(parent,name,mode,dev){return MEMFS.createNode(parent,name,mode,dev)},rename(old_node,new_dir,new_name){var new_node;try{new_node=FS.lookupNode(new_dir,new_name)}catch(e){}if(new_node){if(FS.isDir(old_node.mode)){for(var i in new_node.contents){throw new FS.ErrnoError(55)}}FS.hashRemoveNode(new_node)}delete old_node.parent.contents[old_node.name];new_dir.contents[new_name]=old_node;old_node.name=new_name;new_dir.ctime=new_dir.mtime=old_node.parent.ctime=old_node.parent.mtime=Date.now()},unlink(parent,name){delete parent.contents[name];parent.ctime=parent.mtime=Date.now()},rmdir(parent,name){var node=FS.lookupNode(parent,name);for(var i in node.contents){throw new FS.ErrnoError(55)}delete parent.contents[name];parent.ctime=parent.mtime=Date.now()},readdir(node){return[".","..",...Object.keys(node.contents)]},symlink(parent,newname,oldpath){var node=MEMFS.createNode(parent,newname,511|40960,0);node.link=oldpath;return node},readlink(node){if(!FS.isLink(node.mode)){throw new FS.ErrnoError(28)}return node.link}},stream_ops:{read(stream,buffer,offset,length,position){var contents=stream.node.contents;if(position>=stream.node.usedBytes)return 0;var size=Math.min(stream.node.usedBytes-position,length);if(size>8&&contents.subarray){buffer.set(contents.subarray(position,position+size),offset)}else{for(var i=0;i<size;i++)buffer[offset+i]=contents[position+i]}return size},write(stream,buffer,offset,length,position,canOwn){if(buffer.buffer===(growMemViews(),HEAP8).buffer){canOwn=false}if(!length)return 0;var node=stream.node;node.mtime=node.ctime=Date.now();if(buffer.subarray&&(!node.contents||node.contents.subarray)){if(canOwn){node.contents=buffer.subarray(offset,offset+length);node.usedBytes=length;return length}else if(node.usedBytes===0&&position===0){node.contents=buffer.slice(offset,offset+length);node.usedBytes=length;return length}else if(position+length<=node.usedBytes){node.contents.set(buffer.subarray(offset,offset+length),position);return length}}MEMFS.expandFileStorage(node,position+length);if(node.contents.subarray&&buffer.subarray){node.contents.set(buffer.subarray(offset,offset+length),position)}else{for(var i=0;i<length;i++){node.contents[position+i]=buffer[offset+i]}}node.usedBytes=Math.max(node.usedBytes,position+length);return length},llseek(stream,offset,whence){var position=offset;if(whence===1){position+=stream.position}else if(whence===2){if(FS.isFile(stream.node.mode)){position+=stream.node.usedBytes}}if(position<0){throw new FS.ErrnoError(28)}return position},mmap(stream,length,position,prot,flags){if(!FS.isFile(stream.node.mode)){throw new FS.ErrnoError(43)}var ptr;var allocated;var contents=stream.node.contents;if(!(flags&2)&&contents&&contents.buffer===(growMemViews(),HEAP8).buffer){allocated=false;ptr=contents.byteOffset}else{allocated=true;ptr=mmapAlloc(length);if(!ptr){throw new FS.ErrnoError(48)}if(contents){if(position>0||position+length<contents.length){if(contents.subarray){contents=contents.subarray(position,position+length)}else{contents=Array.prototype.slice.call(contents,position,position+length)}}(growMemViews(),HEAP8).set(contents,ptr)}}return{ptr,allocated}},msync(stream,buffer,offset,length,mmapFlags){MEMFS.stream_ops.write(stream,buffer,0,length,offset,false);return 0}}};var FS_modeStringToFlags=str=>{var flagModes={r:0,"r+":2,w:512|64|1,"w+":512|64|2,a:1024|64|1,"a+":1024|64|2};var flags=flagModes[str];if(typeof flags=="undefined"){throw new Error(`Unknown file open mode: ${str}`)}return flags};var FS_getMode=(canRead,canWrite)=>{var mode=0;if(canRead)mode|=292|73;if(canWrite)mode|=146;return mode};var asyncLoad=async url=>{var arrayBuffer=await readAsync(url);return new Uint8Array(arrayBuffer)};var FS_createDataFile=(...args)=>FS.createDataFile(...args);var getUniqueRunDependency=id=>id;var preloadPlugins=[];var FS_handledByPreloadPlugin=async(byteArray,fullname)=>{if(typeof Browser!="undefined")Browser.init();for(var plugin of preloadPlugins){if(plugin["canHandle"](fullname)){return plugin["handle"](byteArray,fullname)}}return byteArray};var FS_preloadFile=async(parent,name,url,canRead,canWrite,dontCreateFile,canOwn,preFinish)=>{var fullname=name?PATH_FS.resolve(PATH.join2(parent,name)):parent;var dep=getUniqueRunDependency(`cp ${fullname}`);addRunDependency(dep);try{var byteArray=url;if(typeof url=="string"){byteArray=await asyncLoad(url)}byteArray=await FS_handledByPreloadPlugin(byteArray,fullname);preFinish?.();if(!dontCreateFile){FS_createDataFile(parent,name,byteArray,canRead,canWrite,canOwn)}}finally{removeRunDependency(dep)}};var FS_createPreloadedFile=(parent,name,url,canRead,canWrite,onload,onerror,dontCreateFile,canOwn,preFinish)=>{FS_preloadFile(parent,name,url,canRead,canWrite,dontCreateFile,canOwn,preFinish).then(onload).catch(onerror)};var FS={root:null,mounts:[],devices:{},streams:[],nextInode:1,nameTable:null,currentPath:"/",initialized:false,ignorePermissions:true,filesystems:null,syncFSRequests:0,readFiles:{},ErrnoError:class{name="ErrnoError";constructor(errno){this.errno=errno}},FSStream:class{shared={};get object(){return this.node}set object(val){this.node=val}get isRead(){return(this.flags&2097155)!==1}get isWrite(){return(this.flags&2097155)!==0}get isAppend(){return this.flags&1024}get flags(){return this.shared.flags}set flags(val){this.shared.flags=val}get position(){return this.shared.position}set position(val){this.shared.position=val}},FSNode:class{node_ops={};stream_ops={};readMode=292|73;writeMode=146;mounted=null;constructor(parent,name,mode,rdev){if(!parent){parent=this}this.parent=parent;this.mount=parent.mount;this.id=FS.nextInode++;this.name=name;this.mode=mode;this.rdev=rdev;this.atime=this.mtime=this.ctime=Date.now()}get read(){return(this.mode&this.readMode)===this.readMode}set read(val){val?this.mode|=this.readMode:this.mode&=~this.readMode}get write(){return(this.mode&this.writeMode)===this.writeMode}set write(val){val?this.mode|=this.writeMode:this.mode&=~this.writeMode}get isFolder(){return FS.isDir(this.mode)}get isDevice(){return FS.isChrdev(this.mode)}},lookupPath(path,opts={}){if(!path){throw new FS.ErrnoError(44)}opts.follow_mount??=true;if(!PATH.isAbs(path)){path=FS.cwd()+"/"+path}linkloop:for(var nlinks=0;nlinks<40;nlinks++){var parts=path.split("/").filter(p=>!!p);var current=FS.root;var current_path="/";for(var i=0;i<parts.length;i++){var islast=i===parts.length-1;if(islast&&opts.parent){break}if(parts[i]==="."){continue}if(parts[i]===".."){current_path=PATH.dirname(current_path);if(FS.isRoot(current)){path=current_path+"/"+parts.slice(i+1).join("/");nlinks--;continue linkloop}else{current=current.parent}continue}current_path=PATH.join2(current_path,parts[i]);try{current=FS.lookupNode(current,parts[i])}catch(e){if(e?.errno===44&&islast&&opts.noent_okay){return{path:current_path}}throw e}if(FS.isMountpoint(current)&&(!islast||opts.follow_mount)){current=current.mounted.root}if(FS.isLink(current.mode)&&(!islast||opts.follow)){if(!current.node_ops.readlink){throw new FS.ErrnoError(52)}var link=current.node_ops.readlink(current);if(!PATH.isAbs(link)){link=PATH.dirname(current_path)+"/"+link}path=link+"/"+parts.slice(i+1).join("/");continue linkloop}}return{path:current_path,node:current}}throw new FS.ErrnoError(32)},getPath(node){var path;while(true){if(FS.isRoot(node)){var mount=node.mount.mountpoint;if(!path)return mount;return mount[mount.length-1]!=="/"?`${mount}/${path}`:mount+path}path=path?`${node.name}/${path}`:node.name;node=node.parent}},hashName(parentid,name){var hash=0;for(var i=0;i<name.length;i++){hash=(hash<<5)-hash+name.charCodeAt(i)|0}return(parentid+hash>>>0)%FS.nameTable.length},hashAddNode(node){var hash=FS.hashName(node.parent.id,node.name);node.name_next=FS.nameTable[hash];FS.nameTable[hash]=node},hashRemoveNode(node){var hash=FS.hashName(node.parent.id,node.name);if(FS.nameTable[hash]===node){FS.nameTable[hash]=node.name_next}else{var current=FS.nameTable[hash];while(current){if(current.name_next===node){current.name_next=node.name_next;break}current=current.name_next}}},lookupNode(parent,name){var errCode=FS.mayLookup(parent);if(errCode){throw new FS.ErrnoError(errCode)}var hash=FS.hashName(parent.id,name);for(var node=FS.nameTable[hash];node;node=node.name_next){var nodeName=node.name;if(node.parent.id===parent.id&&nodeName===name){return node}}return FS.lookup(parent,name)},createNode(parent,name,mode,rdev){var node=new FS.FSNode(parent,name,mode,rdev);FS.hashAddNode(node);return node},destroyNode(node){FS.hashRemoveNode(node)},isRoot(node){return node===node.parent},isMountpoint(node){return!!node.mounted},isFile(mode){return(mode&61440)===32768},isDir(mode){return(mode&61440)===16384},isLink(mode){return(mode&61440)===40960},isChrdev(mode){return(mode&61440)===8192},isBlkdev(mode){return(mode&61440)===24576},isFIFO(mode){return(mode&61440)===4096},isSocket(mode){return(mode&49152)===49152},flagsToPermissionString(flag){var perms=["r","w","rw"][flag&3];if(flag&512){perms+="w"}return perms},nodePermissions(node,perms){if(FS.ignorePermissions){return 0}if(perms.includes("r")&&!(node.mode&292)){return 2}else if(perms.includes("w")&&!(node.mode&146)){return 2}else if(perms.includes("x")&&!(node.mode&73)){return 2}return 0},mayLookup(dir){if(!FS.isDir(dir.mode))return 54;var errCode=FS.nodePermissions(dir,"x");if(errCode)return errCode;if(!dir.node_ops.lookup)return 2;return 0},mayCreate(dir,name){if(!FS.isDir(dir.mode)){return 54}try{var node=FS.lookupNode(dir,name);return 20}catch(e){}return FS.nodePermissions(dir,"wx")},mayDelete(dir,name,isdir){var node;try{node=FS.lookupNode(dir,name)}catch(e){return e.errno}var errCode=FS.nodePermissions(dir,"wx");if(errCode){return errCode}if(isdir){if(!FS.isDir(node.mode)){return 54}if(FS.isRoot(node)||FS.getPath(node)===FS.cwd()){return 10}}else{if(FS.isDir(node.mode)){return 31}}return 0},mayOpen(node,flags){if(!node){return 44}if(FS.isLink(node.mode)){return 32}else if(FS.isDir(node.mode)){if(FS.flagsToPermissionString(flags)!=="r"||flags&(512|64)){return 31}}return FS.nodePermissions(node,FS.flagsToPermissionString(flags))},checkOpExists(op,err){if(!op){throw new FS.ErrnoError(err)}return op},MAX_OPEN_FDS:4096,nextfd(){for(var fd=0;fd<=FS.MAX_OPEN_FDS;fd++){if(!FS.streams[fd]){return fd}}throw new FS.ErrnoError(33)},getStreamChecked(fd){var stream=FS.getStream(fd);if(!stream){throw new FS.ErrnoError(8)}return stream},getStream:fd=>FS.streams[fd],createStream(stream,fd=-1){stream=Object.assign(new FS.FSStream,stream);if(fd==-1){fd=FS.nextfd()}stream.fd=fd;FS.streams[fd]=stream;return stream},closeStream(fd){FS.streams[fd]=null},dupStream(origStream,fd=-1){var stream=FS.createStream(origStream,fd);stream.stream_ops?.dup?.(stream);return stream},doSetAttr(stream,node,attr){var setattr=stream?.stream_ops.setattr;var arg=setattr?stream:node;setattr??=node.node_ops.setattr;FS.checkOpExists(setattr,63);setattr(arg,attr)},chrdev_stream_ops:{open(stream){var device=FS.getDevice(stream.node.rdev);stream.stream_ops=device.stream_ops;stream.stream_ops.open?.(stream)},llseek(){throw new FS.ErrnoError(70)}},major:dev=>dev>>8,minor:dev=>dev&255,makedev:(ma,mi)=>ma<<8|mi,registerDevice(dev,ops){FS.devices[dev]={stream_ops:ops}},getDevice:dev=>FS.devices[dev],getMounts(mount){var mounts=[];var check=[mount];while(check.length){var m=check.pop();mounts.push(m);check.push(...m.mounts)}return mounts},syncfs(populate,callback){if(typeof populate=="function"){callback=populate;populate=false}FS.syncFSRequests++;if(FS.syncFSRequests>1){err(`warning: ${FS.syncFSRequests} FS.syncfs operations in flight at once, probably just doing extra work`)}var mounts=FS.getMounts(FS.root.mount);var completed=0;function doCallback(errCode){FS.syncFSRequests--;return callback(errCode)}function done(errCode){if(errCode){if(!done.errored){done.errored=true;return doCallback(errCode)}return}if(++completed>=mounts.length){doCallback(null)}}for(var mount of mounts){if(mount.type.syncfs){mount.type.syncfs(mount,populate,done)}else{done(null)}}},mount(type,opts,mountpoint){var root=mountpoint==="/";var pseudo=!mountpoint;var node;if(root&&FS.root){throw new FS.ErrnoError(10)}else if(!root&&!pseudo){var lookup=FS.lookupPath(mountpoint,{follow_mount:false});mountpoint=lookup.path;node=lookup.node;if(FS.isMountpoint(node)){throw new FS.ErrnoError(10)}if(!FS.isDir(node.mode)){throw new FS.ErrnoError(54)}}var mount={type,opts,mountpoint,mounts:[]};var mountRoot=type.mount(mount);mountRoot.mount=mount;mount.root=mountRoot;if(root){FS.root=mountRoot}else if(node){node.mounted=mount;if(node.mount){node.mount.mounts.push(mount)}}return mountRoot},unmount(mountpoint){var lookup=FS.lookupPath(mountpoint,{follow_mount:false});if(!FS.isMountpoint(lookup.node)){throw new FS.ErrnoError(28)}var node=lookup.node;var mount=node.mounted;var mounts=FS.getMounts(mount);for(var[hash,current]of Object.entries(FS.nameTable)){while(current){var next=current.name_next;if(mounts.includes(current.mount)){FS.destroyNode(current)}current=next}}node.mounted=null;var idx=node.mount.mounts.indexOf(mount);node.mount.mounts.splice(idx,1)},lookup(parent,name){return parent.node_ops.lookup(parent,name)},mknod(path,mode,dev){var lookup=FS.lookupPath(path,{parent:true});var parent=lookup.node;var name=PATH.basename(path);if(!name){throw new FS.ErrnoError(28)}if(name==="."||name===".."){throw new FS.ErrnoError(20)}var errCode=FS.mayCreate(parent,name);if(errCode){throw new FS.ErrnoError(errCode)}if(!parent.node_ops.mknod){throw new FS.ErrnoError(63)}return parent.node_ops.mknod(parent,name,mode,dev)},statfs(path){return FS.statfsNode(FS.lookupPath(path,{follow:true}).node)},statfsStream(stream){return FS.statfsNode(stream.node)},statfsNode(node){var rtn={bsize:4096,frsize:4096,blocks:1e6,bfree:5e5,bavail:5e5,files:FS.nextInode,ffree:FS.nextInode-1,fsid:42,flags:2,namelen:255};if(node.node_ops.statfs){Object.assign(rtn,node.node_ops.statfs(node.mount.opts.root))}return rtn},create(path,mode=438){mode&=4095;mode|=32768;return FS.mknod(path,mode,0)},mkdir(path,mode=511){mode&=511|512;mode|=16384;return FS.mknod(path,mode,0)},mkdirTree(path,mode){var dirs=path.split("/");var d="";for(var dir of dirs){if(!dir)continue;if(d||PATH.isAbs(path))d+="/";d+=dir;try{FS.mkdir(d,mode)}catch(e){if(e.errno!=20)throw e}}},mkdev(path,mode,dev){if(typeof dev=="undefined"){dev=mode;mode=438}mode|=8192;return FS.mknod(path,mode,dev)},symlink(oldpath,newpath){if(!PATH_FS.resolve(oldpath)){throw new FS.ErrnoError(44)}var lookup=FS.lookupPath(newpath,{parent:true});var parent=lookup.node;if(!parent){throw new FS.ErrnoError(44)}var newname=PATH.basename(newpath);var errCode=FS.mayCreate(parent,newname);if(errCode){throw new FS.ErrnoError(errCode)}if(!parent.node_ops.symlink){throw new FS.ErrnoError(63)}return parent.node_ops.symlink(parent,newname,oldpath)},rename(old_path,new_path){var old_dirname=PATH.dirname(old_path);var new_dirname=PATH.dirname(new_path);var old_name=PATH.basename(old_path);var new_name=PATH.basename(new_path);var lookup,old_dir,new_dir;lookup=FS.lookupPath(old_path,{parent:true});old_dir=lookup.node;lookup=FS.lookupPath(new_path,{parent:true});new_dir=lookup.node;if(!old_dir||!new_dir)throw new FS.ErrnoError(44);if(old_dir.mount!==new_dir.mount){throw new FS.ErrnoError(75)}var old_node=FS.lookupNode(old_dir,old_name);var relative=PATH_FS.relative(old_path,new_dirname);if(relative.charAt(0)!=="."){throw new FS.ErrnoError(28)}relative=PATH_FS.relative(new_path,old_dirname);if(relative.charAt(0)!=="."){throw new FS.ErrnoError(55)}var new_node;try{new_node=FS.lookupNode(new_dir,new_name)}catch(e){}if(old_node===new_node){return}var isdir=FS.isDir(old_node.mode);var errCode=FS.mayDelete(old_dir,old_name,isdir);if(errCode){throw new FS.ErrnoError(errCode)}errCode=new_node?FS.mayDelete(new_dir,new_name,isdir):FS.mayCreate(new_dir,new_name);if(errCode){throw new FS.ErrnoError(errCode)}if(!old_dir.node_ops.rename){throw new FS.ErrnoError(63)}if(FS.isMountpoint(old_node)||new_node&&FS.isMountpoint(new_node)){throw new FS.ErrnoError(10)}if(new_dir!==old_dir){errCode=FS.nodePermissions(old_dir,"w");if(errCode){throw new FS.ErrnoError(errCode)}}FS.hashRemoveNode(old_node);try{old_dir.node_ops.rename(old_node,new_dir,new_name);old_node.parent=new_dir}catch(e){throw e}finally{FS.hashAddNode(old_node)}},rmdir(path){var lookup=FS.lookupPath(path,{parent:true});var parent=lookup.node;var name=PATH.basename(path);var node=FS.lookupNode(parent,name);var errCode=FS.mayDelete(parent,name,true);if(errCode){throw new FS.ErrnoError(errCode)}if(!parent.node_ops.rmdir){throw new FS.ErrnoError(63)}if(FS.isMountpoint(node)){throw new FS.ErrnoError(10)}parent.node_ops.rmdir(parent,name);FS.destroyNode(node)},readdir(path){var lookup=FS.lookupPath(path,{follow:true});var node=lookup.node;var readdir=FS.checkOpExists(node.node_ops.readdir,54);return readdir(node)},unlink(path){var lookup=FS.lookupPath(path,{parent:true});var parent=lookup.node;if(!parent){throw new FS.ErrnoError(44)}var name=PATH.basename(path);var node=FS.lookupNode(parent,name);var errCode=FS.mayDelete(parent,name,false);if(errCode){throw new FS.ErrnoError(errCode)}if(!parent.node_ops.unlink){throw new FS.ErrnoError(63)}if(FS.isMountpoint(node)){throw new FS.ErrnoError(10)}parent.node_ops.unlink(parent,name);FS.destroyNode(node)},readlink(path){var lookup=FS.lookupPath(path);var link=lookup.node;if(!link){throw new FS.ErrnoError(44)}if(!link.node_ops.readlink){throw new FS.ErrnoError(28)}return link.node_ops.readlink(link)},stat(path,dontFollow){var lookup=FS.lookupPath(path,{follow:!dontFollow});var node=lookup.node;var getattr=FS.checkOpExists(node.node_ops.getattr,63);return getattr(node)},fstat(fd){var stream=FS.getStreamChecked(fd);var node=stream.node;var getattr=stream.stream_ops.getattr;var arg=getattr?stream:node;getattr??=node.node_ops.getattr;FS.checkOpExists(getattr,63);return getattr(arg)},lstat(path){return FS.stat(path,true)},doChmod(stream,node,mode,dontFollow){FS.doSetAttr(stream,node,{mode:mode&4095|node.mode&~4095,ctime:Date.now(),dontFollow})},chmod(path,mode,dontFollow){var node;if(typeof path=="string"){var lookup=FS.lookupPath(path,{follow:!dontFollow});node=lookup.node}else{node=path}FS.doChmod(null,node,mode,dontFollow)},lchmod(path,mode){FS.chmod(path,mode,true)},fchmod(fd,mode){var stream=FS.getStreamChecked(fd);FS.doChmod(stream,stream.node,mode,false)},doChown(stream,node,dontFollow){FS.doSetAttr(stream,node,{timestamp:Date.now(),dontFollow})},chown(path,uid,gid,dontFollow){var node;if(typeof path=="string"){var lookup=FS.lookupPath(path,{follow:!dontFollow});node=lookup.node}else{node=path}FS.doChown(null,node,dontFollow)},lchown(path,uid,gid){FS.chown(path,uid,gid,true)},fchown(fd,uid,gid){var stream=FS.getStreamChecked(fd);FS.doChown(stream,stream.node,false)},doTruncate(stream,node,len){if(FS.isDir(node.mode)){throw new FS.ErrnoError(31)}if(!FS.isFile(node.mode)){throw new FS.ErrnoError(28)}var errCode=FS.nodePermissions(node,"w");if(errCode){throw new FS.ErrnoError(errCode)}FS.doSetAttr(stream,node,{size:len,timestamp:Date.now()})},truncate(path,len){if(len<0){throw new FS.ErrnoError(28)}var node;if(typeof path=="string"){var lookup=FS.lookupPath(path,{follow:true});node=lookup.node}else{node=path}FS.doTruncate(null,node,len)},ftruncate(fd,len){var stream=FS.getStreamChecked(fd);if(len<0||(stream.flags&2097155)===0){throw new FS.ErrnoError(28)}FS.doTruncate(stream,stream.node,len)},utime(path,atime,mtime){var lookup=FS.lookupPath(path,{follow:true});var node=lookup.node;var setattr=FS.checkOpExists(node.node_ops.setattr,63);setattr(node,{atime,mtime})},open(path,flags,mode=438){if(path===""){throw new FS.ErrnoError(44)}flags=typeof flags=="string"?FS_modeStringToFlags(flags):flags;if(flags&64){mode=mode&4095|32768}else{mode=0}var node;var isDirPath;if(typeof path=="object"){node=path}else{isDirPath=path.endsWith("/");var lookup=FS.lookupPath(path,{follow:!(flags&131072),noent_okay:true});node=lookup.node;path=lookup.path}var created=false;if(flags&64){if(node){if(flags&128){throw new FS.ErrnoError(20)}}else if(isDirPath){throw new FS.ErrnoError(31)}else{node=FS.mknod(path,mode|511,0);created=true}}if(!node){throw new FS.ErrnoError(44)}if(FS.isChrdev(node.mode)){flags&=~512}if(flags&65536&&!FS.isDir(node.mode)){throw new FS.ErrnoError(54)}if(!created){var errCode=FS.mayOpen(node,flags);if(errCode){throw new FS.ErrnoError(errCode)}}if(flags&512&&!created){FS.truncate(node,0)}flags&=~(128|512|131072);var stream=FS.createStream({node,path:FS.getPath(node),flags,seekable:true,position:0,stream_ops:node.stream_ops,ungotten:[],error:false});if(stream.stream_ops.open){stream.stream_ops.open(stream)}if(created){FS.chmod(node,mode&511)}if(Module["logReadFiles"]&&!(flags&1)){if(!(path in FS.readFiles)){FS.readFiles[path]=1}}return stream},close(stream){if(FS.isClosed(stream)){throw new FS.ErrnoError(8)}if(stream.getdents)stream.getdents=null;try{if(stream.stream_ops.close){stream.stream_ops.close(stream)}}catch(e){throw e}finally{FS.closeStream(stream.fd)}stream.fd=null},isClosed(stream){return stream.fd===null},llseek(stream,offset,whence){if(FS.isClosed(stream)){throw new FS.ErrnoError(8)}if(!stream.seekable||!stream.stream_ops.llseek){throw new FS.ErrnoError(70)}if(whence!=0&&whence!=1&&whence!=2){throw new FS.ErrnoError(28)}stream.position=stream.stream_ops.llseek(stream,offset,whence);stream.ungotten=[];return stream.position},read(stream,buffer,offset,length,position){if(length<0||position<0){throw new FS.ErrnoError(28)}if(FS.isClosed(stream)){throw new FS.ErrnoError(8)}if((stream.flags&2097155)===1){throw new FS.ErrnoError(8)}if(FS.isDir(stream.node.mode)){throw new FS.ErrnoError(31)}if(!stream.stream_ops.read){throw new FS.ErrnoError(28)}var seeking=typeof position!="undefined";if(!seeking){position=stream.position}else if(!stream.seekable){throw new FS.ErrnoError(70)}var bytesRead=stream.stream_ops.read(stream,buffer,offset,length,position);if(!seeking)stream.position+=bytesRead;return bytesRead},write(stream,buffer,offset,length,position,canOwn){if(length<0||position<0){throw new FS.ErrnoError(28)}if(FS.isClosed(stream)){throw new FS.ErrnoError(8)}if((stream.flags&2097155)===0){throw new FS.ErrnoError(8)}if(FS.isDir(stream.node.mode)){throw new FS.ErrnoError(31)}if(!stream.stream_ops.write){throw new FS.ErrnoError(28)}if(stream.seekable&&stream.flags&1024){FS.llseek(stream,0,2)}var seeking=typeof position!="undefined";if(!seeking){position=stream.position}else if(!stream.seekable){throw new FS.ErrnoError(70)}var bytesWritten=stream.stream_ops.write(stream,buffer,offset,length,position,canOwn);if(!seeking)stream.position+=bytesWritten;return bytesWritten},mmap(stream,length,position,prot,flags){if((prot&2)!==0&&(flags&2)===0&&(stream.flags&2097155)!==2){throw new FS.ErrnoError(2)}if((stream.flags&2097155)===1){throw new FS.ErrnoError(2)}if(!stream.stream_ops.mmap){throw new FS.ErrnoError(43)}if(!length){throw new FS.ErrnoError(28)}return stream.stream_ops.mmap(stream,length,position,prot,flags)},msync(stream,buffer,offset,length,mmapFlags){if(!stream.stream_ops.msync){return 0}return stream.stream_ops.msync(stream,buffer,offset,length,mmapFlags)},ioctl(stream,cmd,arg){if(!stream.stream_ops.ioctl){throw new FS.ErrnoError(59)}return stream.stream_ops.ioctl(stream,cmd,arg)},readFile(path,opts={}){opts.flags=opts.flags||0;opts.encoding=opts.encoding||"binary";if(opts.encoding!=="utf8"&&opts.encoding!=="binary"){abort(`Invalid encoding type "${opts.encoding}"`)}var stream=FS.open(path,opts.flags);var stat=FS.stat(path);var length=stat.size;var buf=new Uint8Array(length);FS.read(stream,buf,0,length,0);if(opts.encoding==="utf8"){buf=UTF8ArrayToString(buf)}FS.close(stream);return buf},writeFile(path,data,opts={}){opts.flags=opts.flags||577;var stream=FS.open(path,opts.flags,opts.mode);if(typeof data=="string"){data=new Uint8Array(intArrayFromString(data,true))}if(ArrayBuffer.isView(data)){FS.write(stream,data,0,data.byteLength,undefined,opts.canOwn)}else{abort("Unsupported data type")}FS.close(stream)},cwd:()=>FS.currentPath,chdir(path){var lookup=FS.lookupPath(path,{follow:true});if(lookup.node===null){throw new FS.ErrnoError(44)}if(!FS.isDir(lookup.node.mode)){throw new FS.ErrnoError(54)}var errCode=FS.nodePermissions(lookup.node,"x");if(errCode){throw new FS.ErrnoError(errCode)}FS.currentPath=lookup.path},createDefaultDirectories(){FS.mkdir("/tmp");FS.mkdir("/home");FS.mkdir("/home/web_user")},createDefaultDevices(){FS.mkdir("/dev");FS.registerDevice(FS.makedev(1,3),{read:()=>0,write:(stream,buffer,offset,length,pos)=>length,llseek:()=>0});FS.mkdev("/dev/null",FS.makedev(1,3));TTY.register(FS.makedev(5,0),TTY.default_tty_ops);TTY.register(FS.makedev(6,0),TTY.default_tty1_ops);FS.mkdev("/dev/tty",FS.makedev(5,0));FS.mkdev("/dev/tty1",FS.makedev(6,0));var randomBuffer=new Uint8Array(1024),randomLeft=0;var randomByte=()=>{if(randomLeft===0){randomFill(randomBuffer);randomLeft=randomBuffer.byteLength}return randomBuffer[--randomLeft]};FS.createDevice("/dev","random",randomByte);FS.createDevice("/dev","urandom",randomByte);FS.mkdir("/dev/shm");FS.mkdir("/dev/shm/tmp")},createSpecialDirectories(){FS.mkdir("/proc");var proc_self=FS.mkdir("/proc/self");FS.mkdir("/proc/self/fd");FS.mount({mount(){var node=FS.createNode(proc_self,"fd",16895,73);node.stream_ops={llseek:MEMFS.stream_ops.llseek};node.node_ops={lookup(parent,name){var fd=+name;var stream=FS.getStreamChecked(fd);var ret={parent:null,mount:{mountpoint:"fake"},node_ops:{readlink:()=>stream.path},id:fd+1};ret.parent=ret;return ret},readdir(){return Array.from(FS.streams.entries()).filter(([k,v])=>v).map(([k,v])=>k.toString())}};return node}},{},"/proc/self/fd")},createStandardStreams(input,output,error){if(input){FS.createDevice("/dev","stdin",input)}else{FS.symlink("/dev/tty","/dev/stdin")}if(output){FS.createDevice("/dev","stdout",null,output)}else{FS.symlink("/dev/tty","/dev/stdout")}if(error){FS.createDevice("/dev","stderr",null,error)}else{FS.symlink("/dev/tty1","/dev/stderr")}var stdin=FS.open("/dev/stdin",0);var stdout=FS.open("/dev/stdout",1);var stderr=FS.open("/dev/stderr",1)},staticInit(){FS.nameTable=new Array(4096);FS.mount(MEMFS,{},"/");FS.createDefaultDirectories();FS.createDefaultDevices();FS.createSpecialDirectories();FS.filesystems={MEMFS}},init(input,output,error){FS.initialized=true;input??=Module["stdin"];output??=Module["stdout"];error??=Module["stderr"];FS.createStandardStreams(input,output,error)},quit(){FS.initialized=false;for(var stream of FS.streams){if(stream){FS.close(stream)}}},findObject(path,dontResolveLastLink){var ret=FS.analyzePath(path,dontResolveLastLink);if(!ret.exists){return null}return ret.object},analyzePath(path,dontResolveLastLink){try{var lookup=FS.lookupPath(path,{follow:!dontResolveLastLink});path=lookup.path}catch(e){}var ret={isRoot:false,exists:false,error:0,name:null,path:null,object:null,parentExists:false,parentPath:null,parentObject:null};try{var lookup=FS.lookupPath(path,{parent:true});ret.parentExists=true;ret.parentPath=lookup.path;ret.parentObject=lookup.node;ret.name=PATH.basename(path);lookup=FS.lookupPath(path,{follow:!dontResolveLastLink});ret.exists=true;ret.path=lookup.path;ret.object=lookup.node;ret.name=lookup.node.name;ret.isRoot=lookup.path==="/"}catch(e){ret.error=e.errno}return ret},createPath(parent,path,canRead,canWrite){parent=typeof parent=="string"?parent:FS.getPath(parent);var parts=path.split("/").reverse();while(parts.length){var part=parts.pop();if(!part)continue;var current=PATH.join2(parent,part);try{FS.mkdir(current)}catch(e){if(e.errno!=20)throw e}parent=current}return current},createFile(parent,name,properties,canRead,canWrite){var path=PATH.join2(typeof parent=="string"?parent:FS.getPath(parent),name);var mode=FS_getMode(canRead,canWrite);return FS.create(path,mode)},createDataFile(parent,name,data,canRead,canWrite,canOwn){var path=name;if(parent){parent=typeof parent=="string"?parent:FS.getPath(parent);path=name?PATH.join2(parent,name):parent}var mode=FS_getMode(canRead,canWrite);var node=FS.create(path,mode);if(data){if(typeof data=="string"){var arr=new Array(data.length);for(var i=0,len=data.length;i<len;++i)arr[i]=data.charCodeAt(i);data=arr}FS.chmod(node,mode|146);var stream=FS.open(node,577);FS.write(stream,data,0,data.length,0,canOwn);FS.close(stream);FS.chmod(node,mode)}},createDevice(parent,name,input,output){var path=PATH.join2(typeof parent=="string"?parent:FS.getPath(parent),name);var mode=FS_getMode(!!input,!!output);FS.createDevice.major??=64;var dev=FS.makedev(FS.createDevice.major++,0);FS.registerDevice(dev,{open(stream){stream.seekable=false},close(stream){if(output?.buffer?.length){output(10)}},read(stream,buffer,offset,length,pos){var bytesRead=0;for(var i=0;i<length;i++){var result;try{result=input()}catch(e){throw new FS.ErrnoError(29)}if(result===undefined&&bytesRead===0){throw new FS.ErrnoError(6)}if(result===null||result===undefined)break;bytesRead++;buffer[offset+i]=result}if(bytesRead){stream.node.atime=Date.now()}return bytesRead},write(stream,buffer,offset,length,pos){for(var i=0;i<length;i++){try{output(buffer[offset+i])}catch(e){throw new FS.ErrnoError(29)}}if(length){stream.node.mtime=stream.node.ctime=Date.now()}return i}});return FS.mkdev(path,mode,dev)},forceLoadFile(obj){if(obj.isDevice||obj.isFolder||obj.link||obj.contents)return true;if(globalThis.XMLHttpRequest){abort("Lazy loading should have been performed (contents set) in createLazyFile, but it was not. Lazy loading only works in web workers. Use --embed-file or --preload-file in emcc on the main thread.")}else{try{obj.contents=readBinary(obj.url)}catch(e){throw new FS.ErrnoError(29)}}},createLazyFile(parent,name,url,canRead,canWrite){class LazyUint8Array{lengthKnown=false;chunks=[];get(idx){if(idx>this.length-1||idx<0){return undefined}var chunkOffset=idx%this.chunkSize;var chunkNum=idx/this.chunkSize|0;return this.getter(chunkNum)[chunkOffset]}setDataGetter(getter){this.getter=getter}cacheLength(){var xhr=new XMLHttpRequest;xhr.open("HEAD",url,false);xhr.send(null);if(!(xhr.status>=200&&xhr.status<300||xhr.status===304))abort("Couldn\'t load "+url+". Status: "+xhr.status);var datalength=Number(xhr.getResponseHeader("Content-length"));var header;var hasByteServing=(header=xhr.getResponseHeader("Accept-Ranges"))&&header==="bytes";var usesGzip=(header=xhr.getResponseHeader("Content-Encoding"))&&header==="gzip";var chunkSize=1024*1024;if(!hasByteServing)chunkSize=datalength;var doXHR=(from,to)=>{if(from>to)abort("invalid range ("+from+", "+to+") or no bytes requested!");if(to>datalength-1)abort("only "+datalength+" bytes available! programmer error!");var xhr=new XMLHttpRequest;xhr.open("GET",url,false);if(datalength!==chunkSize)xhr.setRequestHeader("Range","bytes="+from+"-"+to);xhr.responseType="arraybuffer";if(xhr.overrideMimeType){xhr.overrideMimeType("text/plain; charset=x-user-defined")}xhr.send(null);if(!(xhr.status>=200&&xhr.status<300||xhr.status===304))abort("Couldn\'t load "+url+". Status: "+xhr.status);if(xhr.response!==undefined){return new Uint8Array(xhr.response||[])}return intArrayFromString(xhr.responseText||"",true)};var lazyArray=this;lazyArray.setDataGetter(chunkNum=>{var start=chunkNum*chunkSize;var end=(chunkNum+1)*chunkSize-1;end=Math.min(end,datalength-1);if(typeof lazyArray.chunks[chunkNum]=="undefined"){lazyArray.chunks[chunkNum]=doXHR(start,end)}if(typeof lazyArray.chunks[chunkNum]=="undefined")abort("doXHR failed!");return lazyArray.chunks[chunkNum]});if(usesGzip||!datalength){chunkSize=datalength=1;datalength=this.getter(0).length;chunkSize=datalength;out("LazyFiles on gzip forces download of the whole file when length is accessed")}this._length=datalength;this._chunkSize=chunkSize;this.lengthKnown=true}get length(){if(!this.lengthKnown){this.cacheLength()}return this._length}get chunkSize(){if(!this.lengthKnown){this.cacheLength()}return this._chunkSize}}if(globalThis.XMLHttpRequest){if(!ENVIRONMENT_IS_WORKER)abort("Cannot do synchronous binary XHRs outside webworkers in modern browsers. Use --embed-file or --preload-file in emcc");var lazyArray=new LazyUint8Array;var properties={isDevice:false,contents:lazyArray}}else{var properties={isDevice:false,url}}var node=FS.createFile(parent,name,properties,canRead,canWrite);if(properties.contents){node.contents=properties.contents}else if(properties.url){node.contents=null;node.url=properties.url}Object.defineProperties(node,{usedBytes:{get:function(){return this.contents.length}}});var stream_ops={};for(const[key,fn]of Object.entries(node.stream_ops)){stream_ops[key]=(...args)=>{FS.forceLoadFile(node);return fn(...args)}}function writeChunks(stream,buffer,offset,length,position){var contents=stream.node.contents;if(position>=contents.length)return 0;var size=Math.min(contents.length-position,length);if(contents.slice){for(var i=0;i<size;i++){buffer[offset+i]=contents[position+i]}}else{for(var i=0;i<size;i++){buffer[offset+i]=contents.get(position+i)}}return size}stream_ops.read=(stream,buffer,offset,length,position)=>{FS.forceLoadFile(node);return writeChunks(stream,buffer,offset,length,position)};stream_ops.mmap=(stream,length,position,prot,flags)=>{FS.forceLoadFile(node);var ptr=mmapAlloc(length);if(!ptr){throw new FS.ErrnoError(48)}writeChunks(stream,(growMemViews(),HEAP8),ptr,length,position);return{ptr,allocated:true}};node.stream_ops=stream_ops;return node}};var UTF8ToString=(ptr,maxBytesToRead,ignoreNul)=>ptr?UTF8ArrayToString((growMemViews(),HEAPU8),ptr,maxBytesToRead,ignoreNul):"";var SYSCALLS={DEFAULT_POLLMASK:5,calculateAt(dirfd,path,allowEmpty){if(PATH.isAbs(path)){return path}var dir;if(dirfd===-100){dir=FS.cwd()}else{var dirstream=SYSCALLS.getStreamFromFD(dirfd);dir=dirstream.path}if(path.length==0){if(!allowEmpty){throw new FS.ErrnoError(44)}return dir}return dir+"/"+path},writeStat(buf,stat){(growMemViews(),HEAPU32)[buf/4]=stat.dev;(growMemViews(),HEAPU32)[(buf+4)/4]=stat.mode;(growMemViews(),HEAPU64)[(buf+8)/8]=BigInt(stat.nlink);(growMemViews(),HEAPU32)[(buf+16)/4]=stat.uid;(growMemViews(),HEAPU32)[(buf+20)/4]=stat.gid;(growMemViews(),HEAPU32)[(buf+24)/4]=stat.rdev;(growMemViews(),HEAP64)[(buf+32)/8]=BigInt(stat.size);(growMemViews(),HEAP32)[(buf+40)/4]=4096;(growMemViews(),HEAP32)[(buf+44)/4]=stat.blocks;var atime=stat.atime.getTime();var mtime=stat.mtime.getTime();var ctime=stat.ctime.getTime();(growMemViews(),HEAP64)[(buf+48)/8]=BigInt(Math.floor(atime/1e3));(growMemViews(),HEAPU64)[(buf+56)/8]=BigInt(atime%1e3*1e3*1e3);(growMemViews(),HEAP64)[(buf+64)/8]=BigInt(Math.floor(mtime/1e3));(growMemViews(),HEAPU64)[(buf+72)/8]=BigInt(mtime%1e3*1e3*1e3);(growMemViews(),HEAP64)[(buf+80)/8]=BigInt(Math.floor(ctime/1e3));(growMemViews(),HEAPU64)[(buf+88)/8]=BigInt(ctime%1e3*1e3*1e3);(growMemViews(),HEAP64)[(buf+96)/8]=BigInt(stat.ino);return 0},writeStatFs(buf,stats){(growMemViews(),HEAPU32)[(buf+8)/4]=stats.bsize;(growMemViews(),HEAPU32)[(buf+72)/4]=stats.bsize;(growMemViews(),HEAP64)[(buf+16)/8]=BigInt(stats.blocks);(growMemViews(),HEAP64)[(buf+24)/8]=BigInt(stats.bfree);(growMemViews(),HEAP64)[(buf+32)/8]=BigInt(stats.bavail);(growMemViews(),HEAP64)[(buf+40)/8]=BigInt(stats.files);(growMemViews(),HEAP64)[(buf+48)/8]=BigInt(stats.ffree);(growMemViews(),HEAPU32)[(buf+56)/4]=stats.fsid;(growMemViews(),HEAPU32)[(buf+80)/4]=stats.flags;(growMemViews(),HEAPU32)[(buf+64)/4]=stats.namelen},doMsync(addr,stream,len,flags,offset){if(!FS.isFile(stream.node.mode)){throw new FS.ErrnoError(43)}if(flags&2){return 0}var buffer=(growMemViews(),HEAPU8).slice(addr,addr+len);FS.msync(stream,buffer,offset,len,flags)},getStreamFromFD(fd){var stream=FS.getStreamChecked(fd);return stream},varargs:undefined,getStr(ptr){var ret=UTF8ToString(ptr);return ret}};function ___syscall_fcntl64(fd,cmd,varargs){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(3,0,1,fd,cmd,varargs);varargs=bigintToI53Checked(varargs);SYSCALLS.varargs=varargs;try{var stream=SYSCALLS.getStreamFromFD(fd);switch(cmd){case 0:{var arg=syscallGetVarargI();if(arg<0){return-28}while(FS.streams[arg]){arg++}var newStream;newStream=FS.dupStream(stream,arg);return newStream.fd}case 1:case 2:return 0;case 3:return stream.flags;case 4:{var arg=syscallGetVarargI();stream.flags|=arg;return 0}case 5:{var arg=syscallGetVarargP();var offset=0;(growMemViews(),HEAP16)[(arg+offset)/2]=2;return 0}case 6:case 7:return 0}return-28}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_fstat64(fd,buf){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(4,0,1,fd,buf);buf=bigintToI53Checked(buf);try{return SYSCALLS.writeStat(buf,FS.fstat(fd))}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}var stringToUTF8=(str,outPtr,maxBytesToWrite)=>stringToUTF8Array(str,(growMemViews(),HEAPU8),outPtr,maxBytesToWrite);function ___syscall_getcwd(buf,size){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(5,0,1,buf,size);buf=bigintToI53Checked(buf);size=bigintToI53Checked(size);try{if(size===0)return-28;var cwd=FS.cwd();var cwdLengthInBytes=lengthBytesUTF8(cwd)+1;if(size<cwdLengthInBytes)return-68;stringToUTF8(cwd,buf,size);return cwdLengthInBytes}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_getdents64(fd,dirp,count){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(6,0,1,fd,dirp,count);dirp=bigintToI53Checked(dirp);count=bigintToI53Checked(count);try{var stream=SYSCALLS.getStreamFromFD(fd);stream.getdents||=FS.readdir(stream.path);var struct_size=280;var pos=0;var off=FS.llseek(stream,0,1);var startIdx=Math.floor(off/struct_size);var endIdx=Math.min(stream.getdents.length,startIdx+Math.floor(count/struct_size));for(var idx=startIdx;idx<endIdx;idx++){var id;var type;var name=stream.getdents[idx];if(name==="."){id=stream.node.id;type=4}else if(name===".."){var lookup=FS.lookupPath(stream.path,{parent:true});id=lookup.node.id;type=4}else{var child;try{child=FS.lookupNode(stream.node,name)}catch(e){if(e?.errno===28){continue}throw e}id=child.id;type=FS.isChrdev(child.mode)?2:FS.isDir(child.mode)?4:FS.isLink(child.mode)?10:8}(growMemViews(),HEAP64)[(dirp+pos)/8]=BigInt(id);(growMemViews(),HEAP64)[(dirp+pos+8)/8]=BigInt((idx+1)*struct_size);(growMemViews(),HEAP16)[(dirp+pos+16)/2]=280;(growMemViews(),HEAP8)[dirp+pos+18]=type;stringToUTF8(name,dirp+pos+19,256);pos+=struct_size}FS.llseek(stream,idx*struct_size,0);return pos}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_ioctl(fd,op,varargs){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(7,0,1,fd,op,varargs);varargs=bigintToI53Checked(varargs);SYSCALLS.varargs=varargs;try{var stream=SYSCALLS.getStreamFromFD(fd);switch(op){case 21509:{if(!stream.tty)return-59;return 0}case 21505:{if(!stream.tty)return-59;if(stream.tty.ops.ioctl_tcgets){var termios=stream.tty.ops.ioctl_tcgets(stream);var argp=syscallGetVarargP();(growMemViews(),HEAP32)[argp/4]=termios.c_iflag||0;(growMemViews(),HEAP32)[(argp+4)/4]=termios.c_oflag||0;(growMemViews(),HEAP32)[(argp+8)/4]=termios.c_cflag||0;(growMemViews(),HEAP32)[(argp+12)/4]=termios.c_lflag||0;for(var i=0;i<32;i++){(growMemViews(),HEAP8)[argp+i+17]=termios.c_cc[i]||0}return 0}return 0}case 21510:case 21511:case 21512:{if(!stream.tty)return-59;return 0}case 21506:case 21507:case 21508:{if(!stream.tty)return-59;if(stream.tty.ops.ioctl_tcsets){var argp=syscallGetVarargP();var c_iflag=(growMemViews(),HEAP32)[argp/4];var c_oflag=(growMemViews(),HEAP32)[(argp+4)/4];var c_cflag=(growMemViews(),HEAP32)[(argp+8)/4];var c_lflag=(growMemViews(),HEAP32)[(argp+12)/4];var c_cc=[];for(var i=0;i<32;i++){c_cc.push((growMemViews(),HEAP8)[argp+i+17])}return stream.tty.ops.ioctl_tcsets(stream.tty,op,{c_iflag,c_oflag,c_cflag,c_lflag,c_cc})}return 0}case 21519:{if(!stream.tty)return-59;var argp=syscallGetVarargP();(growMemViews(),HEAP32)[argp/4]=0;return 0}case 21520:{if(!stream.tty)return-59;return-28}case 21537:case 21531:{var argp=syscallGetVarargP();return FS.ioctl(stream,op,argp)}case 21523:{if(!stream.tty)return-59;if(stream.tty.ops.ioctl_tiocgwinsz){var winsize=stream.tty.ops.ioctl_tiocgwinsz(stream.tty);var argp=syscallGetVarargP();(growMemViews(),HEAP16)[argp/2]=winsize[0];(growMemViews(),HEAP16)[(argp+2)/2]=winsize[1]}return 0}case 21524:{if(!stream.tty)return-59;return 0}case 21515:{if(!stream.tty)return-59;return 0}default:return-28}}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_lstat64(path,buf){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(8,0,1,path,buf);path=bigintToI53Checked(path);buf=bigintToI53Checked(buf);try{path=SYSCALLS.getStr(path);return SYSCALLS.writeStat(buf,FS.lstat(path))}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_newfstatat(dirfd,path,buf,flags){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(9,0,1,dirfd,path,buf,flags);path=bigintToI53Checked(path);buf=bigintToI53Checked(buf);try{path=SYSCALLS.getStr(path);var nofollow=flags&256;var allowEmpty=flags&4096;flags=flags&~6400;path=SYSCALLS.calculateAt(dirfd,path,allowEmpty);return SYSCALLS.writeStat(buf,nofollow?FS.lstat(path):FS.stat(path))}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_openat(dirfd,path,flags,varargs){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(10,0,1,dirfd,path,flags,varargs);path=bigintToI53Checked(path);varargs=bigintToI53Checked(varargs);SYSCALLS.varargs=varargs;try{path=SYSCALLS.getStr(path);path=SYSCALLS.calculateAt(dirfd,path);var mode=varargs?syscallGetVarargI():0;return FS.open(path,flags,mode).fd}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_stat64(path,buf){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(11,0,1,path,buf);path=bigintToI53Checked(path);buf=bigintToI53Checked(buf);try{path=SYSCALLS.getStr(path);return SYSCALLS.writeStat(buf,FS.stat(path))}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}var __abort_js=()=>abort("");function __emscripten_init_main_thread_js(tb){tb=bigintToI53Checked(tb);__emscripten_thread_init(tb,!ENVIRONMENT_IS_WORKER,1,!ENVIRONMENT_IS_WEB,5242880,false);PThread.threadInitTLS()}var handleException=e=>{if(e instanceof ExitStatus||e=="unwind"){return EXITSTATUS}quit_(1,e)};var maybeExit=()=>{if(!keepRuntimeAlive()){try{if(ENVIRONMENT_IS_PTHREAD){if(_pthread_self())__emscripten_thread_exit(EXITSTATUS);return}_exit(EXITSTATUS)}catch(e){handleException(e)}}};var callUserCallback=func=>{if(ABORT){return}try{func();maybeExit()}catch(e){handleException(e)}};function __emscripten_thread_mailbox_await(pthread_ptr){pthread_ptr=bigintToI53Checked(pthread_ptr);if(Atomics.waitAsync){var wait=Atomics.waitAsync((growMemViews(),HEAP32),pthread_ptr/4,pthread_ptr);wait.value.then(checkMailbox);var waitingAsync=pthread_ptr+228;Atomics.store((growMemViews(),HEAP32),waitingAsync/4,1)}}var checkMailbox=()=>callUserCallback(()=>{var pthread_ptr=_pthread_self();if(pthread_ptr){__emscripten_thread_mailbox_await(pthread_ptr);__emscripten_check_mailbox()}});function __emscripten_notify_mailbox_postmessage(targetThread,currThreadId){targetThread=bigintToI53Checked(targetThread);currThreadId=bigintToI53Checked(currThreadId);if(targetThread==currThreadId){setTimeout(checkMailbox)}else if(ENVIRONMENT_IS_PTHREAD){postMessage({targetThread,cmd:"checkMailbox"})}else{var worker=PThread.pthreads[targetThread];if(!worker){return}worker.postMessage({cmd:"checkMailbox"})}}var proxiedJSCallArgs=[];function __emscripten_receive_on_main_thread_js(funcIndex,emAsmAddr,callingThread,numCallArgs,args){emAsmAddr=bigintToI53Checked(emAsmAddr);callingThread=bigintToI53Checked(callingThread);args=bigintToI53Checked(args);numCallArgs/=2;proxiedJSCallArgs.length=numCallArgs;var b=args/8;for(var i=0;i<numCallArgs;i++){if((growMemViews(),HEAP64)[b+2*i]){proxiedJSCallArgs[i]=(growMemViews(),HEAP64)[b+2*i+1]}else{proxiedJSCallArgs[i]=(growMemViews(),HEAPF64)[b+2*i+1]}}var func=proxiedFunctionTable[funcIndex];PThread.currentProxiedOperationCallerThread=callingThread;var rtn=func(...proxiedJSCallArgs);PThread.currentProxiedOperationCallerThread=0;if(typeof rtn=="bigint"){rtn=bigintToI53Checked(rtn)}return rtn}function __emscripten_thread_cleanup(thread){thread=bigintToI53Checked(thread);if(!ENVIRONMENT_IS_PTHREAD)cleanupThread(thread);else postMessage({cmd:"cleanupThread",thread})}function __emscripten_thread_set_strongref(thread){thread=bigintToI53Checked(thread);if(ENVIRONMENT_IS_NODE){PThread.pthreads[thread].ref()}}var isLeapYear=year=>year%4===0&&(year%100!==0||year%400===0);var MONTH_DAYS_LEAP_CUMULATIVE=[0,31,60,91,121,152,182,213,244,274,305,335];var MONTH_DAYS_REGULAR_CUMULATIVE=[0,31,59,90,120,151,181,212,243,273,304,334];var ydayFromDate=date=>{var leap=isLeapYear(date.getFullYear());var monthDaysCumulative=leap?MONTH_DAYS_LEAP_CUMULATIVE:MONTH_DAYS_REGULAR_CUMULATIVE;var yday=monthDaysCumulative[date.getMonth()]+date.getDate()-1;return yday};function __localtime_js(time,tmPtr){time=bigintToI53Checked(time);tmPtr=bigintToI53Checked(tmPtr);var date=new Date(time*1e3);(growMemViews(),HEAP32)[tmPtr/4]=date.getSeconds();(growMemViews(),HEAP32)[(tmPtr+4)/4]=date.getMinutes();(growMemViews(),HEAP32)[(tmPtr+8)/4]=date.getHours();(growMemViews(),HEAP32)[(tmPtr+12)/4]=date.getDate();(growMemViews(),HEAP32)[(tmPtr+16)/4]=date.getMonth();(growMemViews(),HEAP32)[(tmPtr+20)/4]=date.getFullYear()-1900;(growMemViews(),HEAP32)[(tmPtr+24)/4]=date.getDay();var yday=ydayFromDate(date)|0;(growMemViews(),HEAP32)[(tmPtr+28)/4]=yday;(growMemViews(),HEAP64)[(tmPtr+40)/8]=BigInt(-(date.getTimezoneOffset()*60));var start=new Date(date.getFullYear(),0,1);var summerOffset=new Date(date.getFullYear(),6,1).getTimezoneOffset();var winterOffset=start.getTimezoneOffset();var dst=(summerOffset!=winterOffset&&date.getTimezoneOffset()==Math.min(winterOffset,summerOffset))|0;(growMemViews(),HEAP32)[(tmPtr+32)/4]=dst}function __mmap_js(len,prot,flags,fd,offset,allocated,addr){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(12,0,1,len,prot,flags,fd,offset,allocated,addr);len=bigintToI53Checked(len);offset=bigintToI53Checked(offset);allocated=bigintToI53Checked(allocated);addr=bigintToI53Checked(addr);try{var stream=SYSCALLS.getStreamFromFD(fd);var res=FS.mmap(stream,len,offset,prot,flags);var ptr=res.ptr;(growMemViews(),HEAP32)[allocated/4]=res.allocated;(growMemViews(),HEAPU64)[addr/8]=BigInt(ptr);return 0}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function __munmap_js(addr,len,prot,flags,fd,offset){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(13,0,1,addr,len,prot,flags,fd,offset);addr=bigintToI53Checked(addr);len=bigintToI53Checked(len);offset=bigintToI53Checked(offset);try{var stream=SYSCALLS.getStreamFromFD(fd);if(prot&2){SYSCALLS.doMsync(addr,stream,len,flags,offset)}}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}var __tzset_js=function(timezone,daylight,std_name,dst_name){timezone=bigintToI53Checked(timezone);daylight=bigintToI53Checked(daylight);std_name=bigintToI53Checked(std_name);dst_name=bigintToI53Checked(dst_name);var currentYear=(new Date).getFullYear();var winter=new Date(currentYear,0,1);var summer=new Date(currentYear,6,1);var winterOffset=winter.getTimezoneOffset();var summerOffset=summer.getTimezoneOffset();var stdTimezoneOffset=Math.max(winterOffset,summerOffset);(growMemViews(),HEAPU64)[timezone/8]=BigInt(stdTimezoneOffset*60);(growMemViews(),HEAP32)[daylight/4]=Number(winterOffset!=summerOffset);var extractZone=timezoneOffset=>{var sign=timezoneOffset>=0?"-":"+";var absOffset=Math.abs(timezoneOffset);var hours=String(Math.floor(absOffset/60)).padStart(2,"0");var minutes=String(absOffset%60).padStart(2,"0");return`UTC${sign}${hours}${minutes}`};var winterName=extractZone(winterOffset);var summerName=extractZone(summerOffset);if(summerOffset<winterOffset){stringToUTF8(winterName,std_name,17);stringToUTF8(summerName,dst_name,17)}else{stringToUTF8(winterName,dst_name,17);stringToUTF8(summerName,std_name,17)}};var _emscripten_get_now=()=>performance.timeOrigin+performance.now();var _emscripten_date_now=()=>Date.now();var nowIsMonotonic=1;var checkWasiClock=clock_id=>clock_id>=0&&clock_id<=3;function _clock_time_get(clk_id,ignored_precision,ptime){ignored_precision=bigintToI53Checked(ignored_precision);ptime=bigintToI53Checked(ptime);if(!checkWasiClock(clk_id)){return 28}var now;if(clk_id===0){now=_emscripten_date_now()}else if(nowIsMonotonic){now=_emscripten_get_now()}else{return 52}var nsec=Math.round(now*1e3*1e3);(growMemViews(),HEAP64)[ptime/8]=BigInt(nsec);return 0}var _emscripten_check_blocking_allowed=()=>{};var runtimeKeepalivePush=()=>{runtimeKeepaliveCounter+=1};var _emscripten_exit_with_live_runtime=()=>{runtimeKeepalivePush();throw"unwind"};var jsStackTrace=()=>(new Error).stack.toString();var getCallstack=flags=>{var callstack=jsStackTrace();var lines=callstack.split("\\n");callstack="";var firefoxRe=new RegExp("\\\\s*(.*?)@(.*?):([0-9]+):([0-9]+)");var chromeRe=new RegExp("\\\\s*at (.*?) \\\\((.*):(.*):(.*)\\\\)");for(var line of lines){var symbolName="";var file="";var lineno=0;var column=0;var parts=chromeRe.exec(line);if(parts?.length==5){symbolName=parts[1];file=parts[2];lineno=parts[3];column=parts[4]}else{parts=firefoxRe.exec(line);if(parts?.length>=4){symbolName=parts[1];file=parts[2];lineno=parts[3];column=parts[4]|0}else{callstack+=line+"\\n";continue}}if(symbolName=="_emscripten_log"||symbolName=="_emscripten_get_callstack"){callstack="";continue}if(flags&24){if(flags&64){file=file.substring(file.replace(/\\\\/g,"/").lastIndexOf("/")+1)}callstack+=`    at ${symbolName} (${file}:${lineno}:${column})\\n`}}callstack=callstack.replace(/\\s+$/,"");return callstack};function _emscripten_get_callstack(flags,str,maxbytes){str=bigintToI53Checked(str);var callstack=getCallstack(flags);if(!str||maxbytes<=0){return lengthBytesUTF8(callstack)+1}var bytesWrittenExcludingNull=stringToUTF8(callstack,str,maxbytes);return bytesWrittenExcludingNull+1}var getHeapMax=()=>4294967296;var _emscripten_get_heap_max=()=>BigInt(getHeapMax());var _emscripten_has_asyncify=()=>2;var _emscripten_num_logical_cores=()=>ENVIRONMENT_IS_NODE?require("os").cpus().length:navigator["hardwareConcurrency"];var growMemory=size=>{var oldHeapSize=wasmMemory.buffer.byteLength;var pages=(size-oldHeapSize+65535)/65536|0;try{wasmMemory.grow(BigInt(pages));updateMemoryViews();return 1}catch(e){}};function _emscripten_resize_heap(requestedSize){requestedSize=bigintToI53Checked(requestedSize);var oldSize=(growMemViews(),HEAPU8).length;if(requestedSize<=oldSize){return false}var maxHeapSize=getHeapMax();if(requestedSize>maxHeapSize){return false}for(var cutDown=1;cutDown<=4;cutDown*=2){var overGrownHeapSize=oldSize*(1+.2/cutDown);overGrownHeapSize=Math.min(overGrownHeapSize,requestedSize+100663296);var newSize=Math.min(maxHeapSize,alignMemory(Math.max(requestedSize,overGrownHeapSize),65536));var replacement=growMemory(newSize);if(replacement){return true}}return false}var stringToUTF8OnStack=str=>{var size=lengthBytesUTF8(str)+1;var ret=stackAlloc(size);stringToUTF8(str,ret,size);return ret};var writeI53ToI64=(ptr,num)=>{(growMemViews(),HEAPU32)[ptr/4]=num;var lower=(growMemViews(),HEAPU32)[ptr/4];(growMemViews(),HEAPU32)[(ptr+4)/4]=(num-lower)/4294967296};var stringToNewUTF8=str=>{var size=lengthBytesUTF8(str)+1;var ret=_malloc(size);if(ret)stringToUTF8(str,ret,size);return ret};var readI53FromI64=ptr=>(growMemViews(),HEAPU32)[ptr/4]+(growMemViews(),HEAP32)[(ptr+4)/4]*4294967296;var WebGPU={Internals:{jsObjects:[],jsObjectInsert:(ptr,jsObject)=>{WebGPU.Internals.jsObjects[ptr]=jsObject},bufferOnUnmaps:[],futures:[],futureInsert:(futureId,promise)=>{WebGPU.Internals.futures[futureId]=new Promise(resolve=>promise.finally(()=>resolve(futureId)))}},getJsObject:ptr=>{if(!ptr)return undefined;return WebGPU.Internals.jsObjects[ptr]},importJsAdapter:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateAdapter(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsBindGroup:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateBindGroup(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsBindGroupLayout:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateBindGroupLayout(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsBuffer:(buffer,parentPtr=0)=>{assert(buffer.mapState==="unmapped");var bufferPtr=_emwgpuCreateBuffer(parentPtr);WebGPU.Internals.jsObjectInsert(bufferPtr,buffer);return bufferPtr},importJsCommandBuffer:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateCommandBuffer(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsCommandEncoder:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateCommandEncoder(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsComputePassEncoder:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateComputePassEncoder(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsComputePipeline:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateComputePipeline(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsDevice:(device,parentPtr=0)=>{var queuePtr=_emwgpuCreateQueue(parentPtr);var devicePtr=_emwgpuCreateDevice(parentPtr,queuePtr);WebGPU.Internals.jsObjectInsert(queuePtr,device.queue);WebGPU.Internals.jsObjectInsert(devicePtr,device);return devicePtr},importJsExternalTexture:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateExternalTexture(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsPipelineLayout:(obj,parentPtr=0)=>{var ptr=_emwgpuCreatePipelineLayout(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsQuerySet:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateQuerySet(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsQueue:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateQueue(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsRenderBundle:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateRenderBundle(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsRenderBundleEncoder:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateRenderBundleEncoder(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsRenderPassEncoder:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateRenderPassEncoder(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsRenderPipeline:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateRenderPipeline(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsSampler:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateSampler(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsShaderModule:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateShaderModule(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsSurface:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateSurface(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsTexture:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateTexture(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsTextureView:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateTextureView(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},errorCallback:(callback,type,message,userdata)=>{var sp=stackSave();var messagePtr=stringToUTF8OnStack(message);((a1,a2,a3)=>getWasmTableEntry(callback).call(null,a1,BigInt(a2),BigInt(a3)))(type,BigInt(messagePtr),userdata);stackRestore(sp)},iterateExtensions:(root,handlers)=>{for(var ptr=Number((growMemViews(),HEAPU64)[root/8]);ptr;ptr=Number((growMemViews(),HEAPU64)[ptr/8])){var sType=(growMemViews(),HEAP32)[(ptr+8)/4];var handler=handlers[sType](ptr)}},setStringView:(ptr,data,length)=>{(growMemViews(),HEAPU64)[ptr/8]=BigInt(data);(growMemViews(),HEAPU64)[(ptr+8)/8]=BigInt(length)},makeStringFromStringView:stringViewPtr=>{var ptr=Number((growMemViews(),HEAPU64)[stringViewPtr/8]);var length=Number((growMemViews(),HEAPU64)[(stringViewPtr+8)/8]);return UTF8ToString(ptr,length)},makeStringFromOptionalStringView:stringViewPtr=>{var ptr=Number((growMemViews(),HEAPU64)[stringViewPtr/8]);var length=Number((growMemViews(),HEAPU64)[(stringViewPtr+8)/8]);if(!ptr){if(length===0){return""}return undefined}return UTF8ToString(ptr,length)},makeColor:ptr=>({r:(growMemViews(),HEAPF64)[ptr/8],g:(growMemViews(),HEAPF64)[(ptr+8)/8],b:(growMemViews(),HEAPF64)[(ptr+16)/8],a:(growMemViews(),HEAPF64)[(ptr+24)/8]}),makeExtent3D:ptr=>({width:(growMemViews(),HEAPU32)[ptr/4],height:(growMemViews(),HEAPU32)[(ptr+4)/4],depthOrArrayLayers:(growMemViews(),HEAPU32)[(ptr+8)/4]}),makeOrigin3D:ptr=>({x:(growMemViews(),HEAPU32)[ptr/4],y:(growMemViews(),HEAPU32)[(ptr+4)/4],z:(growMemViews(),HEAPU32)[(ptr+8)/4]}),makeTexelCopyTextureInfo:ptr=>({texture:WebGPU.getJsObject(Number((growMemViews(),HEAPU64)[ptr/8])),mipLevel:(growMemViews(),HEAPU32)[(ptr+8)/4],origin:WebGPU.makeOrigin3D(ptr+12),aspect:WebGPU.TextureAspect[(growMemViews(),HEAP32)[(ptr+24)/4]]}),makeTexelCopyBufferLayout:ptr=>{var bytesPerRow=(growMemViews(),HEAPU32)[(ptr+8)/4];var rowsPerImage=(growMemViews(),HEAPU32)[(ptr+12)/4];return{offset:readI53FromI64(ptr),bytesPerRow:bytesPerRow===4294967295?undefined:bytesPerRow,rowsPerImage:rowsPerImage===4294967295?undefined:rowsPerImage}},makeTexelCopyBufferInfo:ptr=>{var layoutPtr=ptr+0;var bufferCopyView=WebGPU.makeTexelCopyBufferLayout(layoutPtr);bufferCopyView["buffer"]=WebGPU.getJsObject(Number((growMemViews(),HEAPU64)[(ptr+16)/8]));return bufferCopyView},makePassTimestampWrites:ptr=>{if(ptr===0)return undefined;return{querySet:WebGPU.getJsObject(Number((growMemViews(),HEAPU64)[(ptr+8)/8])),beginningOfPassWriteIndex:(growMemViews(),HEAPU32)[(ptr+16)/4],endOfPassWriteIndex:(growMemViews(),HEAPU32)[(ptr+20)/4]}},makePipelineConstants:(constantCount,constantsPtr)=>{if(!constantCount)return;var constants={};for(var i=0;i<constantCount;++i){var entryPtr=constantsPtr+32*i;var key=WebGPU.makeStringFromStringView(entryPtr+8);constants[key]=(growMemViews(),HEAPF64)[(entryPtr+24)/8]}return constants},makePipelineLayout:layoutPtr=>{if(!layoutPtr)return"auto";return WebGPU.getJsObject(layoutPtr)},makeComputeState:ptr=>{if(!ptr)return undefined;var desc={module:WebGPU.getJsObject(Number((growMemViews(),HEAPU64)[(ptr+8)/8])),constants:WebGPU.makePipelineConstants(Number((growMemViews(),HEAPU64)[(ptr+32)/8]),Number((growMemViews(),HEAPU64)[(ptr+40)/8])),entryPoint:WebGPU.makeStringFromOptionalStringView(ptr+16)};return desc},makeComputePipelineDesc:descriptor=>{var desc={label:WebGPU.makeStringFromOptionalStringView(descriptor+8),layout:WebGPU.makePipelineLayout(Number((growMemViews(),HEAPU64)[(descriptor+24)/8])),compute:WebGPU.makeComputeState(descriptor+32)};return desc},makeRenderPipelineDesc:descriptor=>{function makePrimitiveState(psPtr){if(!psPtr)return undefined;return{topology:WebGPU.PrimitiveTopology[(growMemViews(),HEAP32)[(psPtr+8)/4]],stripIndexFormat:WebGPU.IndexFormat[(growMemViews(),HEAP32)[(psPtr+12)/4]],frontFace:WebGPU.FrontFace[(growMemViews(),HEAP32)[(psPtr+16)/4]],cullMode:WebGPU.CullMode[(growMemViews(),HEAP32)[(psPtr+20)/4]],unclippedDepth:!!(growMemViews(),HEAPU32)[(psPtr+24)/4]}}function makeBlendComponent(bdPtr){if(!bdPtr)return undefined;return{operation:WebGPU.BlendOperation[(growMemViews(),HEAP32)[bdPtr/4]],srcFactor:WebGPU.BlendFactor[(growMemViews(),HEAP32)[(bdPtr+4)/4]],dstFactor:WebGPU.BlendFactor[(growMemViews(),HEAP32)[(bdPtr+8)/4]]}}function makeBlendState(bsPtr){if(!bsPtr)return undefined;return{alpha:makeBlendComponent(bsPtr+12),color:makeBlendComponent(bsPtr+0)}}function makeColorState(csPtr){var format=WebGPU.TextureFormat[(growMemViews(),HEAP32)[(csPtr+8)/4]];return format?{format,blend:makeBlendState(Number((growMemViews(),HEAPU64)[(csPtr+16)/8])),writeMask:(growMemViews(),HEAPU32)[(csPtr+24)/4]}:undefined}function makeColorStates(count,csArrayPtr){var states=[];for(var i=0;i<count;++i){states.push(makeColorState(csArrayPtr+32*i))}return states}function makeStencilStateFace(ssfPtr){return{compare:WebGPU.CompareFunction[(growMemViews(),HEAP32)[ssfPtr/4]],failOp:WebGPU.StencilOperation[(growMemViews(),HEAP32)[(ssfPtr+4)/4]],depthFailOp:WebGPU.StencilOperation[(growMemViews(),HEAP32)[(ssfPtr+8)/4]],passOp:WebGPU.StencilOperation[(growMemViews(),HEAP32)[(ssfPtr+12)/4]]}}function makeDepthStencilState(dssPtr){if(!dssPtr)return undefined;return{format:WebGPU.TextureFormat[(growMemViews(),HEAP32)[(dssPtr+8)/4]],depthWriteEnabled:!!(growMemViews(),HEAPU32)[(dssPtr+12)/4],depthCompare:WebGPU.CompareFunction[(growMemViews(),HEAP32)[(dssPtr+16)/4]],stencilFront:makeStencilStateFace(dssPtr+20),stencilBack:makeStencilStateFace(dssPtr+36),stencilReadMask:(growMemViews(),HEAPU32)[(dssPtr+52)/4],stencilWriteMask:(growMemViews(),HEAPU32)[(dssPtr+56)/4],depthBias:(growMemViews(),HEAP32)[(dssPtr+60)/4],depthBiasSlopeScale:(growMemViews(),HEAPF32)[(dssPtr+64)/4],depthBiasClamp:(growMemViews(),HEAPF32)[(dssPtr+68)/4]}}function makeVertexAttribute(vaPtr){return{format:WebGPU.VertexFormat[(growMemViews(),HEAP32)[(vaPtr+8)/4]],offset:readI53FromI64(vaPtr+16),shaderLocation:(growMemViews(),HEAPU32)[(vaPtr+24)/4]}}function makeVertexAttributes(count,vaArrayPtr){var vas=[];for(var i=0;i<count;++i){vas.push(makeVertexAttribute(vaArrayPtr+i*32))}return vas}function makeVertexBuffer(vbPtr){if(!vbPtr)return undefined;var stepMode=WebGPU.VertexStepMode[(growMemViews(),HEAP32)[(vbPtr+8)/4]];var attributeCount=Number((growMemViews(),HEAPU64)[(vbPtr+24)/8]);if(!stepMode&&!attributeCount){return null}return{arrayStride:readI53FromI64(vbPtr+16),stepMode,attributes:makeVertexAttributes(attributeCount,Number((growMemViews(),HEAPU64)[(vbPtr+32)/8]))}}function makeVertexBuffers(count,vbArrayPtr){if(!count)return undefined;var vbs=[];for(var i=0;i<count;++i){vbs.push(makeVertexBuffer(vbArrayPtr+i*40))}return vbs}function makeVertexState(viPtr){if(!viPtr)return undefined;var desc={module:WebGPU.getJsObject(Number((growMemViews(),HEAPU64)[(viPtr+8)/8])),constants:WebGPU.makePipelineConstants(Number((growMemViews(),HEAPU64)[(viPtr+32)/8]),Number((growMemViews(),HEAPU64)[(viPtr+40)/8])),buffers:makeVertexBuffers(Number((growMemViews(),HEAPU64)[(viPtr+48)/8]),Number((growMemViews(),HEAPU64)[(viPtr+56)/8])),entryPoint:WebGPU.makeStringFromOptionalStringView(viPtr+16)};return desc}function makeMultisampleState(msPtr){if(!msPtr)return undefined;return{count:(growMemViews(),HEAPU32)[(msPtr+8)/4],mask:(growMemViews(),HEAPU32)[(msPtr+12)/4],alphaToCoverageEnabled:!!(growMemViews(),HEAPU32)[(msPtr+16)/4]}}function makeFragmentState(fsPtr){if(!fsPtr)return undefined;var desc={module:WebGPU.getJsObject(Number((growMemViews(),HEAPU64)[(fsPtr+8)/8])),constants:WebGPU.makePipelineConstants(Number((growMemViews(),HEAPU64)[(fsPtr+32)/8]),Number((growMemViews(),HEAPU64)[(fsPtr+40)/8])),targets:makeColorStates(Number((growMemViews(),HEAPU64)[(fsPtr+48)/8]),Number((growMemViews(),HEAPU64)[(fsPtr+56)/8])),entryPoint:WebGPU.makeStringFromOptionalStringView(fsPtr+16)};return desc}var desc={label:WebGPU.makeStringFromOptionalStringView(descriptor+8),layout:WebGPU.makePipelineLayout(Number((growMemViews(),HEAPU64)[(descriptor+24)/8])),vertex:makeVertexState(descriptor+32),primitive:makePrimitiveState(descriptor+96),depthStencil:makeDepthStencilState(Number((growMemViews(),HEAPU64)[(descriptor+128)/8])),multisample:makeMultisampleState(descriptor+136),fragment:makeFragmentState(Number((growMemViews(),HEAPU64)[(descriptor+160)/8]))};return desc},fillLimitStruct:(limits,limitsOutPtr)=>{var nextInChainPtr=Number((growMemViews(),HEAPU64)[limitsOutPtr/8]);function setLimitValueU32(name,basePtr,limitOffset,fallbackValue=0){var limitValue=limits[name]??fallbackValue;(growMemViews(),HEAPU32)[(basePtr+limitOffset)/4]=limitValue}function setLimitValueU64(name,basePtr,limitOffset,fallbackValue=0){var limitValue=limits[name]??fallbackValue;writeI53ToI64(basePtr+limitOffset,limitValue)}setLimitValueU32("maxTextureDimension1D",limitsOutPtr,8);setLimitValueU32("maxTextureDimension2D",limitsOutPtr,12);setLimitValueU32("maxTextureDimension3D",limitsOutPtr,16);setLimitValueU32("maxTextureArrayLayers",limitsOutPtr,20);setLimitValueU32("maxBindGroups",limitsOutPtr,24);setLimitValueU32("maxBindGroupsPlusVertexBuffers",limitsOutPtr,28);setLimitValueU32("maxBindingsPerBindGroup",limitsOutPtr,32);setLimitValueU32("maxDynamicUniformBuffersPerPipelineLayout",limitsOutPtr,36);setLimitValueU32("maxDynamicStorageBuffersPerPipelineLayout",limitsOutPtr,40);setLimitValueU32("maxSampledTexturesPerShaderStage",limitsOutPtr,44);setLimitValueU32("maxSamplersPerShaderStage",limitsOutPtr,48);setLimitValueU32("maxStorageBuffersPerShaderStage",limitsOutPtr,52);setLimitValueU32("maxStorageTexturesPerShaderStage",limitsOutPtr,56);setLimitValueU32("maxUniformBuffersPerShaderStage",limitsOutPtr,60);setLimitValueU32("minUniformBufferOffsetAlignment",limitsOutPtr,80);setLimitValueU32("minStorageBufferOffsetAlignment",limitsOutPtr,84);setLimitValueU64("maxUniformBufferBindingSize",limitsOutPtr,64);setLimitValueU64("maxStorageBufferBindingSize",limitsOutPtr,72);setLimitValueU32("maxVertexBuffers",limitsOutPtr,88);setLimitValueU64("maxBufferSize",limitsOutPtr,96);setLimitValueU32("maxVertexAttributes",limitsOutPtr,104);setLimitValueU32("maxVertexBufferArrayStride",limitsOutPtr,108);setLimitValueU32("maxInterStageShaderVariables",limitsOutPtr,112);setLimitValueU32("maxColorAttachments",limitsOutPtr,116);setLimitValueU32("maxColorAttachmentBytesPerSample",limitsOutPtr,120);setLimitValueU32("maxComputeWorkgroupStorageSize",limitsOutPtr,124);setLimitValueU32("maxComputeInvocationsPerWorkgroup",limitsOutPtr,128);setLimitValueU32("maxComputeWorkgroupSizeX",limitsOutPtr,132);setLimitValueU32("maxComputeWorkgroupSizeY",limitsOutPtr,136);setLimitValueU32("maxComputeWorkgroupSizeZ",limitsOutPtr,140);setLimitValueU32("maxComputeWorkgroupsPerDimension",limitsOutPtr,144);setLimitValueU32("maxImmediateSize",limitsOutPtr,148);if(nextInChainPtr!==0){var sType=(growMemViews(),HEAP32)[(nextInChainPtr+8)/4];var compatibilityModeLimitsPtr=nextInChainPtr;setLimitValueU32("maxStorageBuffersInVertexStage",compatibilityModeLimitsPtr,16,limits.maxStorageBuffersPerShaderStage);setLimitValueU32("maxStorageBuffersInFragmentStage",compatibilityModeLimitsPtr,24,limits.maxStorageBuffersPerShaderStage);setLimitValueU32("maxStorageTexturesInVertexStage",compatibilityModeLimitsPtr,20,limits.maxStorageTexturesPerShaderStage);setLimitValueU32("maxStorageTexturesInFragmentStage",compatibilityModeLimitsPtr,28,limits.maxStorageTexturesPerShaderStage)}},fillAdapterInfoStruct:(info,infoStruct)=>{(growMemViews(),HEAPU32)[(infoStruct+88)/4]=info.subgroupMinSize;(growMemViews(),HEAPU32)[(infoStruct+92)/4]=info.subgroupMaxSize;var strs=info.vendor+info.architecture+info.device+info.description;var strPtr=stringToNewUTF8(strs);var vendorLen=lengthBytesUTF8(info.vendor);WebGPU.setStringView(infoStruct+8,strPtr,vendorLen);strPtr+=vendorLen;var architectureLen=lengthBytesUTF8(info.architecture);WebGPU.setStringView(infoStruct+24,strPtr,architectureLen);strPtr+=architectureLen;var deviceLen=lengthBytesUTF8(info.device);WebGPU.setStringView(infoStruct+40,strPtr,deviceLen);strPtr+=deviceLen;var descriptionLen=lengthBytesUTF8(info.description);WebGPU.setStringView(infoStruct+56,strPtr,descriptionLen);strPtr+=descriptionLen;(growMemViews(),HEAP32)[(infoStruct+72)/4]=2;var adapterType=info.isFallbackAdapter?3:4;(growMemViews(),HEAP32)[(infoStruct+76)/4]=adapterType;(growMemViews(),HEAPU32)[(infoStruct+80)/4]=0;(growMemViews(),HEAPU32)[(infoStruct+84)/4]=0},AddressMode:[,"clamp-to-edge","repeat","mirror-repeat"],BlendFactor:[,"zero","one","src","one-minus-src","src-alpha","one-minus-src-alpha","dst","one-minus-dst","dst-alpha","one-minus-dst-alpha","src-alpha-saturated","constant","one-minus-constant","src1","one-minus-src1","src1-alpha","one-minus-src1-alpha"],BlendOperation:[,"add","subtract","reverse-subtract","min","max"],BufferBindingType:[,,"uniform","storage","read-only-storage"],BufferMapState:[,"unmapped","pending","mapped"],CompareFunction:[,"never","less","equal","less-equal","greater","not-equal","greater-equal","always"],CompilationInfoRequestStatus:[,"success","callback-cancelled"],ComponentSwizzle:[,"0","1","r","g","b","a"],CompositeAlphaMode:[,"opaque","premultiplied","unpremultiplied","inherit"],CullMode:[,"none","front","back"],ErrorFilter:[,"validation","out-of-memory","internal"],FeatureLevel:[,"compatibility","core"],FeatureName:{1:"core-features-and-limits",2:"depth-clip-control",3:"depth32float-stencil8",4:"texture-compression-bc",5:"texture-compression-bc-sliced-3d",6:"texture-compression-etc2",7:"texture-compression-astc",8:"texture-compression-astc-sliced-3d",9:"timestamp-query",10:"indirect-first-instance",11:"shader-f16",12:"rg11b10ufloat-renderable",13:"bgra8unorm-storage",14:"float32-filterable",15:"float32-blendable",16:"clip-distances",17:"dual-source-blending",18:"subgroups",19:"texture-formats-tier1",20:"texture-formats-tier2",21:"primitive-index",22:"texture-component-swizzle",327692:"chromium-experimental-unorm16-texture-formats",327729:"chromium-experimental-multi-draw-indirect"},FilterMode:[,"nearest","linear"],FrontFace:[,"ccw","cw"],IndexFormat:[,"uint16","uint32"],InstanceFeatureName:[,"timed-wait-any","shader-source-spirv","multiple-devices-per-adapter"],LoadOp:[,"load","clear"],MipmapFilterMode:[,"nearest","linear"],OptionalBool:["false","true"],PowerPreference:[,"low-power","high-performance"],PredefinedColorSpace:[,"srgb","display-p3"],PrimitiveTopology:[,"point-list","line-list","line-strip","triangle-list","triangle-strip"],QueryType:[,"occlusion","timestamp"],SamplerBindingType:[,,"filtering","non-filtering","comparison"],Status:[,"success","error"],StencilOperation:[,"keep","zero","replace","invert","increment-clamp","decrement-clamp","increment-wrap","decrement-wrap"],StorageTextureAccess:[,,"write-only","read-only","read-write"],StoreOp:[,"store","discard"],SurfaceGetCurrentTextureStatus:[,"success-optimal","success-suboptimal","timeout","outdated","lost","error"],TextureAspect:[,"all","stencil-only","depth-only"],TextureDimension:[,"1d","2d","3d"],TextureFormat:[,"r8unorm","r8snorm","r8uint","r8sint","r16unorm","r16snorm","r16uint","r16sint","r16float","rg8unorm","rg8snorm","rg8uint","rg8sint","r32float","r32uint","r32sint","rg16unorm","rg16snorm","rg16uint","rg16sint","rg16float","rgba8unorm","rgba8unorm-srgb","rgba8snorm","rgba8uint","rgba8sint","bgra8unorm","bgra8unorm-srgb","rgb10a2uint","rgb10a2unorm","rg11b10ufloat","rgb9e5ufloat","rg32float","rg32uint","rg32sint","rgba16unorm","rgba16snorm","rgba16uint","rgba16sint","rgba16float","rgba32float","rgba32uint","rgba32sint","stencil8","depth16unorm","depth24plus","depth24plus-stencil8","depth32float","depth32float-stencil8","bc1-rgba-unorm","bc1-rgba-unorm-srgb","bc2-rgba-unorm","bc2-rgba-unorm-srgb","bc3-rgba-unorm","bc3-rgba-unorm-srgb","bc4-r-unorm","bc4-r-snorm","bc5-rg-unorm","bc5-rg-snorm","bc6h-rgb-ufloat","bc6h-rgb-float","bc7-rgba-unorm","bc7-rgba-unorm-srgb","etc2-rgb8unorm","etc2-rgb8unorm-srgb","etc2-rgb8a1unorm","etc2-rgb8a1unorm-srgb","etc2-rgba8unorm","etc2-rgba8unorm-srgb","eac-r11unorm","eac-r11snorm","eac-rg11unorm","eac-rg11snorm","astc-4x4-unorm","astc-4x4-unorm-srgb","astc-5x4-unorm","astc-5x4-unorm-srgb","astc-5x5-unorm","astc-5x5-unorm-srgb","astc-6x5-unorm","astc-6x5-unorm-srgb","astc-6x6-unorm","astc-6x6-unorm-srgb","astc-8x5-unorm","astc-8x5-unorm-srgb","astc-8x6-unorm","astc-8x6-unorm-srgb","astc-8x8-unorm","astc-8x8-unorm-srgb","astc-10x5-unorm","astc-10x5-unorm-srgb","astc-10x6-unorm","astc-10x6-unorm-srgb","astc-10x8-unorm","astc-10x8-unorm-srgb","astc-10x10-unorm","astc-10x10-unorm-srgb","astc-12x10-unorm","astc-12x10-unorm-srgb","astc-12x12-unorm","astc-12x12-unorm-srgb"],TextureSampleType:[,,"float","unfilterable-float","depth","sint","uint"],TextureViewDimension:[,"1d","2d","2d-array","cube","cube-array","3d"],ToneMappingMode:[,"standard","extended"],VertexFormat:[,"uint8","uint8x2","uint8x4","sint8","sint8x2","sint8x4","unorm8","unorm8x2","unorm8x4","snorm8","snorm8x2","snorm8x4","uint16","uint16x2","uint16x4","sint16","sint16x2","sint16x4","unorm16","unorm16x2","unorm16x4","snorm16","snorm16x2","snorm16x4","float16","float16x2","float16x4","float32","float32x2","float32x3","float32x4","uint32","uint32x2","uint32x3","uint32x4","sint32","sint32x2","sint32x3","sint32x4","unorm10-10-10-2","unorm8x4-bgra"],VertexStepMode:[,"vertex","instance"],WGSLLanguageFeatureName:[,"readonly_and_readwrite_storage_textures","packed_4x8_integer_dot_product","unrestricted_pointer_parameters","pointer_composite_access","uniform_buffer_standard_layout","subgroup_id","texture_and_sampler_let","subgroup_uniformity","texture_formats_tier1"]};var emwgpuStringToInt_DeviceLostReason={undefined:1,unknown:1,destroyed:2};var runtimeKeepalivePop=()=>{runtimeKeepaliveCounter-=1};function _emwgpuAdapterRequestDevice(adapterPtr,futureId,deviceLostFutureId,devicePtr,queuePtr,descriptor){adapterPtr=bigintToI53Checked(adapterPtr);futureId=bigintToI53Checked(futureId);deviceLostFutureId=bigintToI53Checked(deviceLostFutureId);devicePtr=bigintToI53Checked(devicePtr);queuePtr=bigintToI53Checked(queuePtr);descriptor=bigintToI53Checked(descriptor);var adapter=WebGPU.getJsObject(adapterPtr);var desc={};if(descriptor){var requiredFeatureCount=Number((growMemViews(),HEAPU64)[(descriptor+24)/8]);if(requiredFeatureCount){var requiredFeaturesPtr=Number((growMemViews(),HEAPU64)[(descriptor+32)/8]);desc["requiredFeatures"]=Array.from((growMemViews(),HEAPU32).subarray(requiredFeaturesPtr/4,(requiredFeaturesPtr+requiredFeatureCount*4)/4),feature=>WebGPU.FeatureName[feature])}var limitsPtr=Number((growMemViews(),HEAPU64)[(descriptor+40)/8]);if(limitsPtr){var nextInChainPtr=Number((growMemViews(),HEAPU64)[limitsPtr/8]);var requiredLimits={};function setLimitU32IfDefined(name,basePtr,limitOffset,ignoreIfZero=false){var ptr=basePtr+limitOffset;var value=(growMemViews(),HEAPU32)[ptr/4];if(value!=4294967295&&(!ignoreIfZero||value!=0)){requiredLimits[name]=value}}function setLimitU64IfDefined(name,basePtr,limitOffset){var ptr=basePtr+limitOffset;var limitPart1=(growMemViews(),HEAPU32)[ptr/4];var limitPart2=(growMemViews(),HEAPU32)[(ptr+4)/4];if(limitPart1!=4294967295||limitPart2!=4294967295){requiredLimits[name]=readI53FromI64(ptr)}}setLimitU32IfDefined("maxTextureDimension1D",limitsPtr,8);setLimitU32IfDefined("maxTextureDimension2D",limitsPtr,12);setLimitU32IfDefined("maxTextureDimension3D",limitsPtr,16);setLimitU32IfDefined("maxTextureArrayLayers",limitsPtr,20);setLimitU32IfDefined("maxBindGroups",limitsPtr,24);setLimitU32IfDefined("maxBindGroupsPlusVertexBuffers",limitsPtr,28);setLimitU32IfDefined("maxBindingsPerBindGroup",limitsPtr,32);setLimitU32IfDefined("maxDynamicUniformBuffersPerPipelineLayout",limitsPtr,36);setLimitU32IfDefined("maxDynamicStorageBuffersPerPipelineLayout",limitsPtr,40);setLimitU32IfDefined("maxSampledTexturesPerShaderStage",limitsPtr,44);setLimitU32IfDefined("maxSamplersPerShaderStage",limitsPtr,48);setLimitU32IfDefined("maxStorageBuffersPerShaderStage",limitsPtr,52);setLimitU32IfDefined("maxStorageTexturesPerShaderStage",limitsPtr,56);setLimitU32IfDefined("maxUniformBuffersPerShaderStage",limitsPtr,60);setLimitU32IfDefined("minUniformBufferOffsetAlignment",limitsPtr,80);setLimitU32IfDefined("minStorageBufferOffsetAlignment",limitsPtr,84);setLimitU64IfDefined("maxUniformBufferBindingSize",limitsPtr,64);setLimitU64IfDefined("maxStorageBufferBindingSize",limitsPtr,72);setLimitU32IfDefined("maxVertexBuffers",limitsPtr,88);setLimitU64IfDefined("maxBufferSize",limitsPtr,96);setLimitU32IfDefined("maxVertexAttributes",limitsPtr,104);setLimitU32IfDefined("maxVertexBufferArrayStride",limitsPtr,108);setLimitU32IfDefined("maxInterStageShaderVariables",limitsPtr,112);setLimitU32IfDefined("maxColorAttachments",limitsPtr,116);setLimitU32IfDefined("maxColorAttachmentBytesPerSample",limitsPtr,120);setLimitU32IfDefined("maxComputeWorkgroupStorageSize",limitsPtr,124);setLimitU32IfDefined("maxComputeInvocationsPerWorkgroup",limitsPtr,128);setLimitU32IfDefined("maxComputeWorkgroupSizeX",limitsPtr,132);setLimitU32IfDefined("maxComputeWorkgroupSizeY",limitsPtr,136);setLimitU32IfDefined("maxComputeWorkgroupSizeZ",limitsPtr,140);setLimitU32IfDefined("maxComputeWorkgroupsPerDimension",limitsPtr,144);setLimitU32IfDefined("maxImmediateSize",limitsPtr,148,true);if(nextInChainPtr!==0){var sType=(growMemViews(),HEAP32)[(nextInChainPtr+8)/4];var compatibilityModeLimitsPtr=nextInChainPtr;if("maxStorageBuffersInVertexStage"in GPUSupportedLimits.prototype){setLimitU32IfDefined("maxStorageBuffersInVertexStage",compatibilityModeLimitsPtr,16);setLimitU32IfDefined("maxStorageTexturesInVertexStage",compatibilityModeLimitsPtr,20);setLimitU32IfDefined("maxStorageBuffersInFragmentStage",compatibilityModeLimitsPtr,24);setLimitU32IfDefined("maxStorageTexturesInFragmentStage",compatibilityModeLimitsPtr,28)}}desc["requiredLimits"]=requiredLimits}var defaultQueuePtr=Number((growMemViews(),HEAPU64)[(descriptor+48)/8]);if(defaultQueuePtr){var defaultQueueDesc={label:WebGPU.makeStringFromOptionalStringView(defaultQueuePtr+8)};desc["defaultQueue"]=defaultQueueDesc}desc["label"]=WebGPU.makeStringFromOptionalStringView(descriptor+8)}runtimeKeepalivePush();WebGPU.Internals.futureInsert(futureId,adapter.requestDevice(desc).then(device=>{runtimeKeepalivePop();callUserCallback(()=>{WebGPU.Internals.jsObjectInsert(queuePtr,device.queue);WebGPU.Internals.jsObjectInsert(devicePtr,device);devicePtr=BigInt(devicePtr);WebGPU.Internals.futureInsert(deviceLostFutureId,device.lost.then(info=>{callUserCallback(()=>{device.onuncapturederror=ev=>{};var sp=stackSave();var messagePtr=stringToUTF8OnStack(info.message);_emwgpuOnDeviceLostCompleted(deviceLostFutureId,emwgpuStringToInt_DeviceLostReason[info.reason],BigInt(messagePtr));stackRestore(sp)})}));device.onuncapturederror=ev=>{var type=5;if(ev.error instanceof GPUValidationError)type=2;else if(ev.error instanceof GPUOutOfMemoryError)type=3;else if(ev.error instanceof GPUInternalError)type=4;var sp=stackSave();var messagePtr=stringToUTF8OnStack(ev.error.message);_emwgpuOnUncapturedError(BigInt(devicePtr),type,BigInt(messagePtr));stackRestore(sp)};_emwgpuOnRequestDeviceCompleted(futureId,1,BigInt(devicePtr),0n)})},ex=>{runtimeKeepalivePop();callUserCallback(()=>{var sp=stackSave();var messagePtr=stringToUTF8OnStack(ex.message);_emwgpuOnRequestDeviceCompleted(futureId,3,BigInt(devicePtr),BigInt(messagePtr));if(deviceLostFutureId){_emwgpuOnDeviceLostCompleted(deviceLostFutureId,4,BigInt(messagePtr))}stackRestore(sp)})}))}function _emwgpuBufferDestroy(bufferPtr){bufferPtr=bigintToI53Checked(bufferPtr);var buffer=WebGPU.getJsObject(bufferPtr);var onUnmap=WebGPU.Internals.bufferOnUnmaps[bufferPtr];if(onUnmap){for(var i=0;i<onUnmap.length;++i){onUnmap[i]()}delete WebGPU.Internals.bufferOnUnmaps[bufferPtr]}buffer.destroy()}var warnOnce=text=>{warnOnce.shown||={};if(!warnOnce.shown[text]){warnOnce.shown[text]=1;if(ENVIRONMENT_IS_NODE)text="warning: "+text;err(text)}};var _emwgpuBufferGetConstMappedRange=function(bufferPtr,offset,size){bufferPtr=bigintToI53Checked(bufferPtr);offset=bigintToI53Checked(offset);size=bigintToI53Checked(size);var ret=(()=>{var buffer=WebGPU.getJsObject(bufferPtr);if(size==-1)size=undefined;var mapped;try{mapped=buffer.getMappedRange(offset,size)}catch(ex){return 0n}var data=_memalign(16,mapped.byteLength);(growMemViews(),HEAPU8).set(new Uint8Array(mapped),data);WebGPU.Internals.bufferOnUnmaps[bufferPtr].push(()=>_free(data));return data})();return BigInt(ret)};var _emwgpuBufferMapAsync=function(bufferPtr,futureId,mode,offset,size){bufferPtr=bigintToI53Checked(bufferPtr);futureId=bigintToI53Checked(futureId);mode=bigintToI53Checked(mode);offset=bigintToI53Checked(offset);size=bigintToI53Checked(size);var buffer=WebGPU.getJsObject(bufferPtr);WebGPU.Internals.bufferOnUnmaps[bufferPtr]=[];if(size==-1)size=undefined;runtimeKeepalivePush();WebGPU.Internals.futureInsert(futureId,buffer.mapAsync(mode,offset,size).then(()=>{runtimeKeepalivePop();callUserCallback(()=>{_emwgpuOnMapAsyncCompleted(futureId,1,0n)})},ex=>{runtimeKeepalivePop();callUserCallback(()=>{var sp=stackSave();var messagePtr=stringToUTF8OnStack(ex.message);var status=ex.name==="AbortError"?4:ex.name==="OperationError"?3:0;_emwgpuOnMapAsyncCompleted(futureId,status,BigInt(messagePtr));delete WebGPU.Internals.bufferOnUnmaps[bufferPtr]})}))};function _emwgpuBufferUnmap(bufferPtr){bufferPtr=bigintToI53Checked(bufferPtr);var buffer=WebGPU.getJsObject(bufferPtr);var onUnmap=WebGPU.Internals.bufferOnUnmaps[bufferPtr];if(!onUnmap){return}for(var i=0;i<onUnmap.length;++i){onUnmap[i]()}delete WebGPU.Internals.bufferOnUnmaps[bufferPtr];buffer.unmap()}function _emwgpuDelete(ptr){ptr=bigintToI53Checked(ptr);delete WebGPU.Internals.jsObjects[ptr]}function _emwgpuDeviceCreateBuffer(devicePtr,descriptor,bufferPtr){devicePtr=bigintToI53Checked(devicePtr);descriptor=bigintToI53Checked(descriptor);bufferPtr=bigintToI53Checked(bufferPtr);var mappedAtCreation=!!(growMemViews(),HEAPU32)[(descriptor+40)/4];var desc={label:WebGPU.makeStringFromOptionalStringView(descriptor+8),usage:(growMemViews(),HEAPU32)[(descriptor+24)/4],size:readI53FromI64(descriptor+32),mappedAtCreation};var device=WebGPU.getJsObject(devicePtr);var buffer;try{buffer=device.createBuffer(desc)}catch(ex){return false}WebGPU.Internals.jsObjectInsert(bufferPtr,buffer);if(mappedAtCreation){WebGPU.Internals.bufferOnUnmaps[bufferPtr]=[]}return true}function _emwgpuDeviceCreateShaderModule(devicePtr,descriptor,shaderModulePtr){devicePtr=bigintToI53Checked(devicePtr);descriptor=bigintToI53Checked(descriptor);shaderModulePtr=bigintToI53Checked(shaderModulePtr);var nextInChainPtr=Number((growMemViews(),HEAPU64)[descriptor/8]);var sType=(growMemViews(),HEAP32)[(nextInChainPtr+8)/4];var desc={label:WebGPU.makeStringFromOptionalStringView(descriptor+8),code:""};switch(sType){case 2:{desc["code"]=WebGPU.makeStringFromStringView(nextInChainPtr+16);break}}var device=WebGPU.getJsObject(devicePtr);WebGPU.Internals.jsObjectInsert(shaderModulePtr,device.createShaderModule(desc))}var _emwgpuDeviceDestroy=devicePtr=>{const device=WebGPU.getJsObject(devicePtr);device.onuncapturederror=null;device.destroy()};function _emwgpuInstanceRequestAdapter(instancePtr,futureId,options,adapterPtr){instancePtr=bigintToI53Checked(instancePtr);futureId=bigintToI53Checked(futureId);options=bigintToI53Checked(options);adapterPtr=bigintToI53Checked(adapterPtr);var opts;if(options){opts={featureLevel:WebGPU.FeatureLevel[(growMemViews(),HEAP32)[(options+8)/4]],powerPreference:WebGPU.PowerPreference[(growMemViews(),HEAP32)[(options+12)/4]],forceFallbackAdapter:!!(growMemViews(),HEAPU32)[(options+16)/4]};var nextInChainPtr=Number((growMemViews(),HEAPU64)[options/8]);if(nextInChainPtr!==0){var sType=(growMemViews(),HEAP32)[(nextInChainPtr+8)/4];var webxrOptions=nextInChainPtr;opts.xrCompatible=!!(growMemViews(),HEAPU32)[(webxrOptions+16)/4]}}if(!("gpu"in navigator)){var sp=stackSave();var messagePtr=stringToUTF8OnStack("WebGPU not available on this browser (navigator.gpu is not available)");_emwgpuOnRequestAdapterCompleted(futureId,3,BigInt(adapterPtr),BigInt(messagePtr));stackRestore(sp);return}runtimeKeepalivePush();WebGPU.Internals.futureInsert(futureId,navigator.gpu.requestAdapter(opts).then(adapter=>{runtimeKeepalivePop();callUserCallback(()=>{if(adapter){WebGPU.Internals.jsObjectInsert(adapterPtr,adapter);_emwgpuOnRequestAdapterCompleted(futureId,1,BigInt(adapterPtr),0n)}else{var sp=stackSave();var messagePtr=stringToUTF8OnStack("WebGPU not available on this browser (requestAdapter returned null)");_emwgpuOnRequestAdapterCompleted(futureId,3,BigInt(adapterPtr),BigInt(messagePtr));stackRestore(sp)}})},ex=>{runtimeKeepalivePop();callUserCallback(()=>{var sp=stackSave();var messagePtr=stringToUTF8OnStack(ex.message);_emwgpuOnRequestAdapterCompleted(futureId,4,BigInt(adapterPtr),BigInt(messagePtr));stackRestore(sp)})}))}var _emwgpuQueueOnSubmittedWorkDone=function(queuePtr,futureId){queuePtr=bigintToI53Checked(queuePtr);futureId=bigintToI53Checked(futureId);var queue=WebGPU.getJsObject(queuePtr);runtimeKeepalivePush();WebGPU.Internals.futureInsert(futureId,queue.onSubmittedWorkDone().then(()=>{runtimeKeepalivePop();callUserCallback(()=>{_emwgpuOnWorkDoneCompleted(futureId,1)})}))};var _emwgpuWaitAny=function(futurePtr,futureCount,timeoutMSPtr){futurePtr=bigintToI53Checked(futurePtr);futureCount=bigintToI53Checked(futureCount);timeoutMSPtr=bigintToI53Checked(timeoutMSPtr);return Asyncify.handleAsync(async()=>{var promises=[];if(timeoutMSPtr){var timeoutMS=(growMemViews(),HEAP32)[timeoutMSPtr/4];promises.length=futureCount+1;promises[futureCount]=new Promise(resolve=>setTimeout(resolve,timeoutMS,0))}else{promises.length=futureCount}for(var i=0;i<futureCount;++i){var futureId=readI53FromI64(futurePtr+i*8);if(!(futureId in WebGPU.Internals.futures)){return futureId}promises[i]=WebGPU.Internals.futures[futureId]}const firstResolvedFuture=await Promise.race(promises);delete WebGPU.Internals.futures[firstResolvedFuture];return firstResolvedFuture})};_emwgpuWaitAny.isAsync=true;var ENV={};var getExecutableName=()=>thisProgram||"./this.program";var getEnvStrings=()=>{if(!getEnvStrings.strings){var lang=(typeof navigator=="object"&&navigator.language||"C").replace("-","_")+".UTF-8";var env={USER:"web_user",LOGNAME:"web_user",PATH:"/",PWD:"/",HOME:"/home/web_user",LANG:lang,_:getExecutableName()};for(var x in ENV){if(ENV[x]===undefined)delete env[x];else env[x]=ENV[x]}var strings=[];for(var x in env){strings.push(`${x}=${env[x]}`)}getEnvStrings.strings=strings}return getEnvStrings.strings};function _environ_get(__environ,environ_buf){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(14,0,1,__environ,environ_buf);__environ=bigintToI53Checked(__environ);environ_buf=bigintToI53Checked(environ_buf);var bufSize=0;var envp=0;for(var string of getEnvStrings()){var ptr=environ_buf+bufSize;(growMemViews(),HEAPU64)[(__environ+envp)/8]=BigInt(ptr);bufSize+=stringToUTF8(string,ptr,Infinity)+1;envp+=8}return 0}function _environ_sizes_get(penviron_count,penviron_buf_size){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(15,0,1,penviron_count,penviron_buf_size);penviron_count=bigintToI53Checked(penviron_count);penviron_buf_size=bigintToI53Checked(penviron_buf_size);var strings=getEnvStrings();(growMemViews(),HEAPU64)[penviron_count/8]=BigInt(strings.length);var bufSize=0;for(var string of strings){bufSize+=lengthBytesUTF8(string)+1}(growMemViews(),HEAPU64)[penviron_buf_size/8]=BigInt(bufSize);return 0}function _fd_close(fd){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(16,0,1,fd);try{var stream=SYSCALLS.getStreamFromFD(fd);FS.close(stream);return 0}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return e.errno}}var doReadv=(stream,iov,iovcnt,offset)=>{var ret=0;for(var i=0;i<iovcnt;i++){var ptr=Number((growMemViews(),HEAPU64)[iov/8]);var len=Number((growMemViews(),HEAPU64)[(iov+8)/8]);iov+=16;var curr=FS.read(stream,(growMemViews(),HEAP8),ptr,len,offset);if(curr<0)return-1;ret+=curr;if(curr<len)break;if(typeof offset!="undefined"){offset+=curr}}return ret};function _fd_read(fd,iov,iovcnt,pnum){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(17,0,1,fd,iov,iovcnt,pnum);iov=bigintToI53Checked(iov);iovcnt=bigintToI53Checked(iovcnt);pnum=bigintToI53Checked(pnum);try{var stream=SYSCALLS.getStreamFromFD(fd);var num=doReadv(stream,iov,iovcnt);(growMemViews(),HEAPU64)[pnum/8]=BigInt(num);return 0}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return e.errno}}function _fd_seek(fd,offset,whence,newOffset){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(18,0,1,fd,offset,whence,newOffset);offset=bigintToI53Checked(offset);newOffset=bigintToI53Checked(newOffset);try{if(isNaN(offset))return 61;var stream=SYSCALLS.getStreamFromFD(fd);FS.llseek(stream,offset,whence);(growMemViews(),HEAP64)[newOffset/8]=BigInt(stream.position);if(stream.getdents&&offset===0&&whence===0)stream.getdents=null;return 0}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return e.errno}}var doWritev=(stream,iov,iovcnt,offset)=>{var ret=0;for(var i=0;i<iovcnt;i++){var ptr=Number((growMemViews(),HEAPU64)[iov/8]);var len=Number((growMemViews(),HEAPU64)[(iov+8)/8]);iov+=16;var curr=FS.write(stream,(growMemViews(),HEAP8),ptr,len,offset);if(curr<0)return-1;ret+=curr;if(curr<len){break}if(typeof offset!="undefined"){offset+=curr}}return ret};function _fd_write(fd,iov,iovcnt,pnum){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(19,0,1,fd,iov,iovcnt,pnum);iov=bigintToI53Checked(iov);iovcnt=bigintToI53Checked(iovcnt);pnum=bigintToI53Checked(pnum);try{var stream=SYSCALLS.getStreamFromFD(fd);var num=doWritev(stream,iov,iovcnt);(growMemViews(),HEAPU64)[pnum/8]=BigInt(num);return 0}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return e.errno}}function _random_get(buffer,size){buffer=bigintToI53Checked(buffer);size=bigintToI53Checked(size);try{randomFill((growMemViews(),HEAPU8).subarray(buffer,buffer+size));return 0}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return e.errno}}function _wgpuAdapterGetInfo(adapterPtr,info){adapterPtr=bigintToI53Checked(adapterPtr);info=bigintToI53Checked(info);var adapter=WebGPU.getJsObject(adapterPtr);WebGPU.fillAdapterInfoStruct(adapter.info,info);return 1}function _wgpuAdapterGetLimits(adapterPtr,limitsOutPtr){adapterPtr=bigintToI53Checked(adapterPtr);limitsOutPtr=bigintToI53Checked(limitsOutPtr);var adapter=WebGPU.getJsObject(adapterPtr);WebGPU.fillLimitStruct(adapter.limits,limitsOutPtr);return 1}function _wgpuAdapterHasFeature(adapterPtr,featureEnumValue){adapterPtr=bigintToI53Checked(adapterPtr);var adapter=WebGPU.getJsObject(adapterPtr);return adapter.features.has(WebGPU.FeatureName[featureEnumValue])}var _wgpuBufferGetSize=function(bufferPtr){bufferPtr=bigintToI53Checked(bufferPtr);var ret=(()=>{var buffer=WebGPU.getJsObject(bufferPtr);return buffer.size})();return BigInt(ret)};var _wgpuCommandEncoderBeginComputePass=function(encoderPtr,descriptor){encoderPtr=bigintToI53Checked(encoderPtr);descriptor=bigintToI53Checked(descriptor);var ret=(()=>{var desc;if(descriptor){desc={label:WebGPU.makeStringFromOptionalStringView(descriptor+8),timestampWrites:WebGPU.makePassTimestampWrites(Number((growMemViews(),HEAPU64)[(descriptor+24)/8]))}}var commandEncoder=WebGPU.getJsObject(encoderPtr);var ptr=_emwgpuCreateComputePassEncoder(0n);WebGPU.Internals.jsObjectInsert(ptr,commandEncoder.beginComputePass(desc));return ptr})();return BigInt(ret)};function _wgpuCommandEncoderCopyBufferToBuffer(encoderPtr,srcPtr,srcOffset,dstPtr,dstOffset,size){encoderPtr=bigintToI53Checked(encoderPtr);srcPtr=bigintToI53Checked(srcPtr);srcOffset=bigintToI53Checked(srcOffset);dstPtr=bigintToI53Checked(dstPtr);dstOffset=bigintToI53Checked(dstOffset);size=bigintToI53Checked(size);var commandEncoder=WebGPU.getJsObject(encoderPtr);var src=WebGPU.getJsObject(srcPtr);var dst=WebGPU.getJsObject(dstPtr);commandEncoder.copyBufferToBuffer(src,srcOffset,dst,dstOffset,size)}var _wgpuCommandEncoderFinish=function(encoderPtr,descriptor){encoderPtr=bigintToI53Checked(encoderPtr);descriptor=bigintToI53Checked(descriptor);var ret=(()=>{var commandEncoder=WebGPU.getJsObject(encoderPtr);var ptr=_emwgpuCreateCommandBuffer(0n);WebGPU.Internals.jsObjectInsert(ptr,commandEncoder.finish());return ptr})();return BigInt(ret)};function _wgpuComputePassEncoderDispatchWorkgroups(passPtr,x,y,z){passPtr=bigintToI53Checked(passPtr);var pass=WebGPU.getJsObject(passPtr);pass.dispatchWorkgroups(x,y,z)}function _wgpuComputePassEncoderEnd(passPtr){passPtr=bigintToI53Checked(passPtr);var pass=WebGPU.getJsObject(passPtr);pass.end()}function _wgpuComputePassEncoderSetBindGroup(passPtr,groupIndex,groupPtr,dynamicOffsetCount,dynamicOffsetsPtr){passPtr=bigintToI53Checked(passPtr);groupPtr=bigintToI53Checked(groupPtr);dynamicOffsetCount=bigintToI53Checked(dynamicOffsetCount);dynamicOffsetsPtr=bigintToI53Checked(dynamicOffsetsPtr);var pass=WebGPU.getJsObject(passPtr);var group=WebGPU.getJsObject(groupPtr);if(dynamicOffsetCount==0){pass.setBindGroup(groupIndex,group)}else{pass.setBindGroup(groupIndex,group,(growMemViews(),HEAPU32),dynamicOffsetsPtr/4,dynamicOffsetCount)}}function _wgpuComputePassEncoderSetPipeline(passPtr,pipelinePtr){passPtr=bigintToI53Checked(passPtr);pipelinePtr=bigintToI53Checked(pipelinePtr);var pass=WebGPU.getJsObject(passPtr);var pipeline=WebGPU.getJsObject(pipelinePtr);pass.setPipeline(pipeline)}var _wgpuComputePipelineGetBindGroupLayout=function(pipelinePtr,groupIndex){pipelinePtr=bigintToI53Checked(pipelinePtr);var ret=(()=>{var pipeline=WebGPU.getJsObject(pipelinePtr);var ptr=_emwgpuCreateBindGroupLayout(0n);WebGPU.Internals.jsObjectInsert(ptr,pipeline.getBindGroupLayout(groupIndex));return ptr})();return BigInt(ret)};var _wgpuDeviceCreateBindGroup=function(devicePtr,descriptor){devicePtr=bigintToI53Checked(devicePtr);descriptor=bigintToI53Checked(descriptor);var ret=(()=>{function makeEntry(entryPtr){var bufferPtr=Number((growMemViews(),HEAPU64)[(entryPtr+16)/8]);var samplerPtr=Number((growMemViews(),HEAPU64)[(entryPtr+40)/8]);var textureViewPtr=Number((growMemViews(),HEAPU64)[(entryPtr+48)/8]);var externalTexturePtr=0;WebGPU.iterateExtensions(entryPtr,{327681:ptr=>{externalTexturePtr=Number((growMemViews(),HEAPU64)[(ptr+16)/8])}});var resource;if(bufferPtr){var size=readI53FromI64(entryPtr+32);if(size==-1)size=undefined;resource={buffer:WebGPU.getJsObject(bufferPtr),offset:readI53FromI64(entryPtr+24),size}}else{resource=WebGPU.getJsObject(samplerPtr||textureViewPtr||externalTexturePtr)}return{binding:(growMemViews(),HEAPU32)[(entryPtr+8)/4],resource}}function makeEntries(count,entriesPtrs){var entries=[];for(var i=0;i<count;++i){entries.push(makeEntry(entriesPtrs+56*i))}return entries}var desc={label:WebGPU.makeStringFromOptionalStringView(descriptor+8),layout:WebGPU.getJsObject(Number((growMemViews(),HEAPU64)[(descriptor+24)/8])),entries:makeEntries(Number((growMemViews(),HEAPU64)[(descriptor+32)/8]),Number((growMemViews(),HEAPU64)[(descriptor+40)/8]))};var device=WebGPU.getJsObject(devicePtr);var ptr=_emwgpuCreateBindGroup(0n);WebGPU.Internals.jsObjectInsert(ptr,device.createBindGroup(desc));return ptr})();return BigInt(ret)};var _wgpuDeviceCreateCommandEncoder=function(devicePtr,descriptor){devicePtr=bigintToI53Checked(devicePtr);descriptor=bigintToI53Checked(descriptor);var ret=(()=>{var desc;if(descriptor){desc={label:WebGPU.makeStringFromOptionalStringView(descriptor+8)}}var device=WebGPU.getJsObject(devicePtr);var ptr=_emwgpuCreateCommandEncoder(0n);WebGPU.Internals.jsObjectInsert(ptr,device.createCommandEncoder(desc));return ptr})();return BigInt(ret)};var _wgpuDeviceCreateComputePipeline=function(devicePtr,descriptor){devicePtr=bigintToI53Checked(devicePtr);descriptor=bigintToI53Checked(descriptor);var ret=(()=>{var desc=WebGPU.makeComputePipelineDesc(descriptor);var device=WebGPU.getJsObject(devicePtr);var ptr=_emwgpuCreateComputePipeline(0n);WebGPU.Internals.jsObjectInsert(ptr,device.createComputePipeline(desc));return ptr})();return BigInt(ret)};function _wgpuInstanceHasWGSLLanguageFeature(instance,featureEnumValue){instance=bigintToI53Checked(instance);if(!("wgslLanguageFeatures"in navigator.gpu)){return false}return navigator.gpu.wgslLanguageFeatures.has(WebGPU.WGSLLanguageFeatureName[featureEnumValue])}var _wgpuQueueSubmit=function(queuePtr,commandCount,commands){queuePtr=bigintToI53Checked(queuePtr);commandCount=bigintToI53Checked(commandCount);commands=bigintToI53Checked(commands);var queue=WebGPU.getJsObject(queuePtr);var cmds=Array.from((growMemViews(),HEAP64).subarray(commands/8,(commands+commandCount*8)/8),id=>WebGPU.getJsObject(id));queue.submit(cmds)};function _wgpuQueueWriteBuffer(queuePtr,bufferPtr,bufferOffset,data,size){queuePtr=bigintToI53Checked(queuePtr);bufferPtr=bigintToI53Checked(bufferPtr);bufferOffset=bigintToI53Checked(bufferOffset);data=bigintToI53Checked(data);size=bigintToI53Checked(size);var queue=WebGPU.getJsObject(queuePtr);var buffer=WebGPU.getJsObject(bufferPtr);var subarray=(growMemViews(),HEAPU8).subarray(data,data+size);queue.writeBuffer(buffer,bufferOffset,subarray,0,size)}var Asyncify={instrumentWasmImports(imports){var importPattern=/^(invoke_.*|__asyncjs__.*)$/;for(let[x,original]of Object.entries(imports)){if(typeof original=="function"){let isAsyncifyImport=original.isAsync||importPattern.test(x);if(isAsyncifyImport){imports[x]=original=new WebAssembly.Suspending(original)}}}},instrumentFunction(original){var wrapper=(...args)=>original(...args);return wrapper},instrumentWasmExports(exports){var exportPattern=/^(wllama_start|wllama_action|main|__main_argc_argv)$/;Asyncify.asyncExports=new Set;var ret={};for(let[x,original]of Object.entries(exports)){if(typeof original=="function"){let isAsyncifyExport=exportPattern.test(x);if(isAsyncifyExport){Asyncify.asyncExports.add(original);original=Asyncify.makeAsyncFunction(original)}var wrapper=Asyncify.instrumentFunction(original);ret[x]=wrapper}else{ret[x]=original}}return ret},asyncExports:null,isAsyncExport(func){return Asyncify.asyncExports?.has(func)},handleAsync:async startAsync=>{runtimeKeepalivePush();try{return await startAsync()}finally{runtimeKeepalivePop()}},handleSleep:startAsync=>Asyncify.handleAsync(()=>new Promise(startAsync)),makeAsyncFunction(original){return WebAssembly.promising(original)}};var getCFunc=ident=>{var func=Module["_"+ident];return func};var writeArrayToMemory=(array,buffer)=>{(growMemViews(),HEAP8).set(array,buffer)};var ccall=(ident,returnType,argTypes,args,opts)=>{var toC={pointer:p=>BigInt(p),string:str=>{var ret=0;if(str!==null&&str!==undefined&&str!==0){ret=stringToUTF8OnStack(str)}return BigInt(ret)},array:arr=>{var ret=stackAlloc(arr.length);writeArrayToMemory(arr,ret);return BigInt(ret)}};function convertReturnValue(ret){if(returnType==="string"){return UTF8ToString(Number(ret))}if(returnType==="pointer")return Number(ret);if(returnType==="boolean")return Boolean(ret);return ret}var func=getCFunc(ident);var cArgs=[];var stack=0;if(args){for(var i=0;i<args.length;i++){var converter=toC[argTypes[i]];if(converter){if(stack===0)stack=stackSave();cArgs[i]=converter(args[i])}else{cArgs[i]=args[i]}}}var ret=func(...cArgs);function onDone(ret){if(stack!==0)stackRestore(stack);return convertReturnValue(ret)}var asyncMode=opts?.async;if(asyncMode)return ret.then(onDone);ret=onDone(ret);return ret};var cwrap=(ident,returnType,argTypes,opts)=>{var numericArgs=!argTypes||argTypes.every(type=>type==="number"||type==="boolean");var numericRet=returnType!=="string";if(numericRet&&numericArgs&&!opts){return getCFunc(ident)}return(...args)=>ccall(ident,returnType,argTypes,args,opts)};var FS_createPath=(...args)=>FS.createPath(...args);var FS_unlink=(...args)=>FS.unlink(...args);var FS_createLazyFile=(...args)=>FS.createLazyFile(...args);var FS_createDevice=(...args)=>FS.createDevice(...args);PThread.init();FS.createPreloadedFile=FS_createPreloadedFile;FS.preloadFile=FS_preloadFile;FS.staticInit();{initMemory();if(Module["noExitRuntime"])noExitRuntime=Module["noExitRuntime"];if(Module["preloadPlugins"])preloadPlugins=Module["preloadPlugins"];if(Module["print"])out=Module["print"];if(Module["printErr"])err=Module["printErr"];if(Module["wasmBinary"])wasmBinary=Module["wasmBinary"];if(Module["arguments"])arguments_=Module["arguments"];if(Module["thisProgram"])thisProgram=Module["thisProgram"];if(Module["preInit"]){if(typeof Module["preInit"]=="function")Module["preInit"]=[Module["preInit"]];while(Module["preInit"].length>0){Module["preInit"].shift()()}}}Module["ENV"]=ENV;Module["mmapAlloc"]=mmapAlloc;Module["wasmMemory"]=wasmMemory;Module["addRunDependency"]=addRunDependency;Module["removeRunDependency"]=removeRunDependency;Module["ccall"]=ccall;Module["cwrap"]=cwrap;Module["FS_preloadFile"]=FS_preloadFile;Module["FS_unlink"]=FS_unlink;Module["FS_createPath"]=FS_createPath;Module["FS_createDevice"]=FS_createDevice;Module["FS"]=FS;Module["FS_createDataFile"]=FS_createDataFile;Module["FS_createLazyFile"]=FS_createLazyFile;Module["MEMFS"]=MEMFS;var proxiedFunctionTable=[_proc_exit,exitOnMainThread,pthreadCreateProxied,___syscall_fcntl64,___syscall_fstat64,___syscall_getcwd,___syscall_getdents64,___syscall_ioctl,___syscall_lstat64,___syscall_newfstatat,___syscall_openat,___syscall_stat64,__mmap_js,__munmap_js,_environ_get,_environ_sizes_get,_fd_close,_fd_read,_fd_seek,_fd_write];function __asyncjs__js_file_read(path_ptr,offset,req_size,out_ptr){return Asyncify.handleAsync(async()=>await _wllama_js_file_read(UTF8ToString(Number(path_ptr)),Number(offset),Number(req_size),Number(out_ptr)))}__asyncjs__js_file_read.sig="jjjjj";var _wllama_malloc,_wllama_start,_wllama_action,_wllama_exit,_wllama_debug,_main,_malloc,_free,_emwgpuCreateBindGroup,_emwgpuCreateBindGroupLayout,_emwgpuCreateCommandBuffer,_emwgpuCreateCommandEncoder,_emwgpuCreateComputePassEncoder,_emwgpuCreateComputePipeline,_emwgpuCreateExternalTexture,_emwgpuCreatePipelineLayout,_emwgpuCreateQuerySet,_emwgpuCreateRenderBundle,_emwgpuCreateRenderBundleEncoder,_emwgpuCreateRenderPassEncoder,_emwgpuCreateRenderPipeline,_emwgpuCreateSampler,_emwgpuCreateSurface,_emwgpuCreateTexture,_emwgpuCreateTextureView,_emwgpuCreateAdapter,_emwgpuCreateBuffer,_emwgpuCreateDevice,_emwgpuCreateQueue,_emwgpuCreateShaderModule,_emwgpuOnDeviceLostCompleted,_emwgpuOnMapAsyncCompleted,_emwgpuOnRequestAdapterCompleted,_emwgpuOnRequestDeviceCompleted,_emwgpuOnWorkDoneCompleted,_emwgpuOnUncapturedError,__emscripten_tls_init,_pthread_self,_emscripten_builtin_memalign,__emscripten_thread_init,__emscripten_thread_crashed,__emscripten_run_js_on_main_thread,__emscripten_thread_free_data,__emscripten_thread_exit,__emscripten_check_mailbox,_memalign,___trap,_emscripten_stack_set_limits,__emscripten_stack_restore,__emscripten_stack_alloc,_emscripten_stack_get_current,__indirect_function_table,wasmTable;function assignWasmExports(wasmExports){_wllama_malloc=Module["_wllama_malloc"]=wasmExports["wllama_malloc"];_wllama_start=Module["_wllama_start"]=wasmExports["wllama_start"];_wllama_action=Module["_wllama_action"]=wasmExports["wllama_action"];_wllama_exit=Module["_wllama_exit"]=wasmExports["wllama_exit"];_wllama_debug=Module["_wllama_debug"]=wasmExports["wllama_debug"];_main=Module["_main"]=wasmExports["main"];_malloc=wasmExports["malloc"];_free=wasmExports["free"];_emwgpuCreateBindGroup=wasmExports["emwgpuCreateBindGroup"];_emwgpuCreateBindGroupLayout=wasmExports["emwgpuCreateBindGroupLayout"];_emwgpuCreateCommandBuffer=wasmExports["emwgpuCreateCommandBuffer"];_emwgpuCreateCommandEncoder=wasmExports["emwgpuCreateCommandEncoder"];_emwgpuCreateComputePassEncoder=wasmExports["emwgpuCreateComputePassEncoder"];_emwgpuCreateComputePipeline=wasmExports["emwgpuCreateComputePipeline"];_emwgpuCreateExternalTexture=wasmExports["emwgpuCreateExternalTexture"];_emwgpuCreatePipelineLayout=wasmExports["emwgpuCreatePipelineLayout"];_emwgpuCreateQuerySet=wasmExports["emwgpuCreateQuerySet"];_emwgpuCreateRenderBundle=wasmExports["emwgpuCreateRenderBundle"];_emwgpuCreateRenderBundleEncoder=wasmExports["emwgpuCreateRenderBundleEncoder"];_emwgpuCreateRenderPassEncoder=wasmExports["emwgpuCreateRenderPassEncoder"];_emwgpuCreateRenderPipeline=wasmExports["emwgpuCreateRenderPipeline"];_emwgpuCreateSampler=wasmExports["emwgpuCreateSampler"];_emwgpuCreateSurface=wasmExports["emwgpuCreateSurface"];_emwgpuCreateTexture=wasmExports["emwgpuCreateTexture"];_emwgpuCreateTextureView=wasmExports["emwgpuCreateTextureView"];_emwgpuCreateAdapter=wasmExports["emwgpuCreateAdapter"];_emwgpuCreateBuffer=wasmExports["emwgpuCreateBuffer"];_emwgpuCreateDevice=wasmExports["emwgpuCreateDevice"];_emwgpuCreateQueue=wasmExports["emwgpuCreateQueue"];_emwgpuCreateShaderModule=wasmExports["emwgpuCreateShaderModule"];_emwgpuOnDeviceLostCompleted=wasmExports["emwgpuOnDeviceLostCompleted"];_emwgpuOnMapAsyncCompleted=wasmExports["emwgpuOnMapAsyncCompleted"];_emwgpuOnRequestAdapterCompleted=wasmExports["emwgpuOnRequestAdapterCompleted"];_emwgpuOnRequestDeviceCompleted=wasmExports["emwgpuOnRequestDeviceCompleted"];_emwgpuOnWorkDoneCompleted=wasmExports["emwgpuOnWorkDoneCompleted"];_emwgpuOnUncapturedError=wasmExports["emwgpuOnUncapturedError"];__emscripten_tls_init=wasmExports["_emscripten_tls_init"];_pthread_self=wasmExports["pthread_self"];_emscripten_builtin_memalign=wasmExports["emscripten_builtin_memalign"];__emscripten_thread_init=wasmExports["_emscripten_thread_init"];__emscripten_thread_crashed=wasmExports["_emscripten_thread_crashed"];__emscripten_run_js_on_main_thread=wasmExports["_emscripten_run_js_on_main_thread"];__emscripten_thread_free_data=wasmExports["_emscripten_thread_free_data"];__emscripten_thread_exit=wasmExports["_emscripten_thread_exit"];__emscripten_check_mailbox=wasmExports["_emscripten_check_mailbox"];_memalign=wasmExports["memalign"];___trap=wasmExports["__trap"];_emscripten_stack_set_limits=wasmExports["emscripten_stack_set_limits"];__emscripten_stack_restore=wasmExports["_emscripten_stack_restore"];__emscripten_stack_alloc=wasmExports["_emscripten_stack_alloc"];_emscripten_stack_get_current=wasmExports["emscripten_stack_get_current"];__indirect_function_table=wasmTable=wasmExports["__indirect_function_table"]}var wasmImports;function assignWasmImports(){wasmImports={__asyncjs__js_file_read,__pthread_create_js:___pthread_create_js,__syscall_fcntl64:___syscall_fcntl64,__syscall_getcwd:___syscall_getcwd,__syscall_getdents64:___syscall_getdents64,__syscall_ioctl:___syscall_ioctl,__syscall_openat:___syscall_openat,__syscall_stat64:___syscall_stat64,_abort_js:__abort_js,_emscripten_init_main_thread_js:__emscripten_init_main_thread_js,_emscripten_notify_mailbox_postmessage:__emscripten_notify_mailbox_postmessage,_emscripten_receive_on_main_thread_js:__emscripten_receive_on_main_thread_js,_emscripten_thread_cleanup:__emscripten_thread_cleanup,_emscripten_thread_mailbox_await:__emscripten_thread_mailbox_await,_emscripten_thread_set_strongref:__emscripten_thread_set_strongref,_localtime_js:__localtime_js,_mmap_js:__mmap_js,_munmap_js:__munmap_js,_tzset_js:__tzset_js,clock_time_get:_clock_time_get,emscripten_check_blocking_allowed:_emscripten_check_blocking_allowed,emscripten_date_now:_emscripten_date_now,emscripten_exit_with_live_runtime:_emscripten_exit_with_live_runtime,emscripten_get_callstack:_emscripten_get_callstack,emscripten_get_heap_max:_emscripten_get_heap_max,emscripten_get_now:_emscripten_get_now,emscripten_has_asyncify:_emscripten_has_asyncify,emscripten_num_logical_cores:_emscripten_num_logical_cores,emscripten_resize_heap:_emscripten_resize_heap,emwgpuAdapterRequestDevice:_emwgpuAdapterRequestDevice,emwgpuBufferDestroy:_emwgpuBufferDestroy,emwgpuBufferGetConstMappedRange:_emwgpuBufferGetConstMappedRange,emwgpuBufferMapAsync:_emwgpuBufferMapAsync,emwgpuBufferUnmap:_emwgpuBufferUnmap,emwgpuDelete:_emwgpuDelete,emwgpuDeviceCreateBuffer:_emwgpuDeviceCreateBuffer,emwgpuDeviceCreateShaderModule:_emwgpuDeviceCreateShaderModule,emwgpuDeviceDestroy:_emwgpuDeviceDestroy,emwgpuInstanceRequestAdapter:_emwgpuInstanceRequestAdapter,emwgpuQueueOnSubmittedWorkDone:_emwgpuQueueOnSubmittedWorkDone,emwgpuWaitAny:_emwgpuWaitAny,environ_get:_environ_get,environ_sizes_get:_environ_sizes_get,exit:_exit,fd_close:_fd_close,fd_read:_fd_read,fd_seek:_fd_seek,fd_write:_fd_write,memory:wasmMemory,random_get:_random_get,wgpuAdapterGetInfo:_wgpuAdapterGetInfo,wgpuAdapterGetLimits:_wgpuAdapterGetLimits,wgpuAdapterHasFeature:_wgpuAdapterHasFeature,wgpuBufferGetSize:_wgpuBufferGetSize,wgpuCommandEncoderBeginComputePass:_wgpuCommandEncoderBeginComputePass,wgpuCommandEncoderCopyBufferToBuffer:_wgpuCommandEncoderCopyBufferToBuffer,wgpuCommandEncoderFinish:_wgpuCommandEncoderFinish,wgpuComputePassEncoderDispatchWorkgroups:_wgpuComputePassEncoderDispatchWorkgroups,wgpuComputePassEncoderEnd:_wgpuComputePassEncoderEnd,wgpuComputePassEncoderSetBindGroup:_wgpuComputePassEncoderSetBindGroup,wgpuComputePassEncoderSetPipeline:_wgpuComputePassEncoderSetPipeline,wgpuComputePipelineGetBindGroupLayout:_wgpuComputePipelineGetBindGroupLayout,wgpuDeviceCreateBindGroup:_wgpuDeviceCreateBindGroup,wgpuDeviceCreateCommandEncoder:_wgpuDeviceCreateCommandEncoder,wgpuDeviceCreateComputePipeline:_wgpuDeviceCreateComputePipeline,wgpuInstanceHasWGSLLanguageFeature:_wgpuInstanceHasWGSLLanguageFeature,wgpuQueueSubmit:_wgpuQueueSubmit,wgpuQueueWriteBuffer:_wgpuQueueWriteBuffer}}function applySignatureConversions(wasmExports){wasmExports=Object.assign({},wasmExports);var makeWrapper___PP=f=>(a0,a1,a2)=>f(a0,BigInt(a1?a1:0),BigInt(a2?a2:0));var makeWrapper_pp=f=>a0=>Number(f(BigInt(a0)));var makeWrapper__p=f=>a0=>f(BigInt(a0));var makeWrapper_p=f=>()=>Number(f());var makeWrapper_ppp=f=>(a0,a1)=>Number(f(BigInt(a0),BigInt(a1)));var makeWrapper__p_____=f=>(a0,a1,a2,a3,a4,a5)=>f(BigInt(a0),a1,a2,a3,a4,a5);var makeWrapper___p_p_=f=>(a0,a1,a2,a3,a4)=>f(a0,BigInt(a1),a2,BigInt(a3),a4);var makeWrapper__pp=f=>(a0,a1)=>f(BigInt(a0),BigInt(a1));wasmExports["main"]=makeWrapper___PP(wasmExports["main"]);wasmExports["malloc"]=makeWrapper_pp(wasmExports["malloc"]);wasmExports["free"]=makeWrapper__p(wasmExports["free"]);wasmExports["pthread_self"]=makeWrapper_p(wasmExports["pthread_self"]);wasmExports["emscripten_builtin_memalign"]=makeWrapper_ppp(wasmExports["emscripten_builtin_memalign"]);wasmExports["_emscripten_thread_init"]=makeWrapper__p_____(wasmExports["_emscripten_thread_init"]);wasmExports["_emscripten_run_js_on_main_thread"]=makeWrapper___p_p_(wasmExports["_emscripten_run_js_on_main_thread"]);wasmExports["_emscripten_thread_free_data"]=makeWrapper__p(wasmExports["_emscripten_thread_free_data"]);wasmExports["_emscripten_thread_exit"]=makeWrapper__p(wasmExports["_emscripten_thread_exit"]);wasmExports["memalign"]=makeWrapper_ppp(wasmExports["memalign"]);wasmExports["emscripten_stack_set_limits"]=makeWrapper__pp(wasmExports["emscripten_stack_set_limits"]);wasmExports["_emscripten_stack_restore"]=makeWrapper__p(wasmExports["_emscripten_stack_restore"]);wasmExports["_emscripten_stack_alloc"]=makeWrapper_pp(wasmExports["_emscripten_stack_alloc"]);wasmExports["emscripten_stack_get_current"]=makeWrapper_p(wasmExports["emscripten_stack_get_current"]);return wasmExports}async function callMain(){var entryFunction=_main;var argc=0;var argv=0;try{var ret=entryFunction(argc,BigInt(argv));ret=await ret;exitJS(ret,true);return ret}catch(e){return handleException(e)}}function run(){if(runDependencies>0){dependenciesFulfilled=run;return}if(ENVIRONMENT_IS_PTHREAD){initRuntime();return}preRun();if(runDependencies>0){dependenciesFulfilled=run;return}async function doRun(){Module["calledRun"]=true;if(ABORT)return;initRuntime();preMain();Module["onRuntimeInitialized"]?.();var noInitialRun=Module["noInitialRun"]||false;if(!noInitialRun)await callMain();postRun()}if(Module["setStatus"]){Module["setStatus"]("Running...");setTimeout(()=>{setTimeout(()=>Module["setStatus"](""),1);doRun()},1)}else{doRun()}}var wasmExports;if(!ENVIRONMENT_IS_PTHREAD){createWasm();run()}\n';

// src/worker.ts
var FILE_READ_REQ_EVENT = "fs.read_req";
var JSPI_STUB = `
if (!WebAssembly.Suspending) {
  // JSPI not available - stubs that keep the import/export tables valid.
  // Suspending wraps imports: identity is fine since async imports won't be called.
  WebAssembly.Suspending = function (fn) {
    // console.log(fn.toString());
    return fn;
  };
  // promising wraps exports: must return a Promise so ccall's ret.then() works.
  WebAssembly.promising = function (fn) {
    return function (...args) {
      try {
        return Promise.resolve(fn(...args));
      } catch (e) {
        return Promise.reject(e);
      }
    };
  };
}
`;
var ProxyToWorker = class {
  // filename -> Blob for async reads
  constructor(resources, nbThread, suppressNativeLog, logger) {
    __publicField(this, "resources");
    __publicField(this, "logger");
    __publicField(this, "suppressNativeLog");
    __publicField(this, "taskQueue", []);
    __publicField(this, "taskId", 1);
    __publicField(this, "resultQueue", []);
    __publicField(this, "busy", false);
    // is the work loop is running?
    __publicField(this, "worker");
    __publicField(this, "multiThread");
    __publicField(this, "nbThread");
    __publicField(this, "useAsyncFile");
    __publicField(this, "fileBlobs", /* @__PURE__ */ new Map());
    this.resources = resources;
    this.nbThread = nbThread;
    this.multiThread = nbThread > 0;
    this.logger = logger;
    this.suppressNativeLog = suppressNativeLog;
    this.useAsyncFile = canUseAsyncFileRead(resources.compat);
  }
  getModuleCode() {
    return __async(this, null, function* () {
      if (!this.resources.jsPath) {
        if (this.resources.compat) {
          throw new Error(
            "compat mode is enabled but no jsPath was provided. Pass a worker JS via setCompat() or install @wllama/wllama-compat."
          );
        }
        return WLLAMA_EMSCRIPTEN_CODE;
      } else if (this.resources.jsPath.code) {
        return this.resources.jsPath.code;
      } else if (isString(this.resources.jsPath)) {
        const response = yield fetch(this.resources.jsPath);
        if (!response.ok) {
          throw new Error(
            `Failed to fetch worker code from ${this.resources.jsPath}`
          );
        }
        return yield response.text();
      } else {
        throw new Error("No JS code provided for worker");
      }
    });
  }
  moduleInit(ggufFiles) {
    return __async(this, null, function* () {
      let moduleCode = JSPI_STUB + (yield this.getModuleCode());
      if (this.resources.noWebGPU) {
        moduleCode = 'try{Object.defineProperty(WorkerNavigator.prototype,"gpu",{get:()=>({requestAdapter:async()=>null})});}catch(e){}' + moduleCode;
      }
      let mainModuleCode = moduleCode.replace("var Module", "var ___Module");
      const runOptions = {
        pathConfig: {
          "wllama.wasm": this.resources.wasmPath
        },
        nbThread: this.nbThread,
        compat: this.resources.compat
      };
      const completeCode = [
        `const RUN_OPTIONS = ${JSON.stringify(runOptions)};`,
        `function wModuleInit() { ${mainModuleCode}; return Module; }`,
        LLAMA_CPP_WORKER_CODE
      ].join(";\n\n");
      this.worker = createWorker(completeCode);
      this.worker.onmessage = this.onRecvMsg.bind(this);
      this.worker.onerror = this.logger.error;
      const res = yield this.pushTask({
        verb: "module.init",
        args: [
          new Blob([moduleCode], { type: "text/javascript" }),
          this.useAsyncFile
        ],
        callbackId: this.taskId++
      });
      const nativeFiles = [];
      for (const file of ggufFiles) {
        const needAllocBuffer = !this.useAsyncFile;
        const id = yield this.fileAlloc(
          file.name,
          file.blob.size,
          needAllocBuffer
        );
        nativeFiles.push(__spreadValues({ id }, file));
        if (this.useAsyncFile) {
          this.fileBlobs.set(file.name, file.blob);
        }
      }
      if (!this.useAsyncFile) {
        yield Promise.all(
          nativeFiles.map((file) => {
            return this.fileWrite(file.id, file.blob);
          })
        );
      }
      return res;
    });
  }
  wllamaStart() {
    return __async(this, null, function* () {
      const result = yield this.pushTask({
        verb: "wllama.start",
        args: [],
        callbackId: this.taskId++
      });
      const parsedResult = this.parseResult(result);
      return parsedResult;
    });
  }
  wllamaAction(name, body) {
    return __async(this, null, function* () {
      const encodedMsg = glueSerialize(body);
      const result = yield this.pushTask({
        verb: "wllama.action",
        args: [name, encodedMsg],
        callbackId: this.taskId++
      });
      const parsedResult = glueDeserialize(result);
      return parsedResult;
    });
  }
  wllamaExit() {
    return __async(this, null, function* () {
      if (this.worker) {
        this.worker.terminate();
      }
    });
  }
  wllamaDebug() {
    return __async(this, null, function* () {
      const result = yield this.pushTask({
        verb: "wllama.debug",
        args: [],
        callbackId: this.taskId++
      });
      return JSON.parse(result);
    });
  }
  ///////////////////////////////////////
  /**
   * Allocate a new file in heapfs
   * @returns fileId, to be used by fileWrite()
   */
  fileAlloc(fileName, size, allocBuffer) {
    return __async(this, null, function* () {
      const result = yield this.pushTask({
        verb: "fs.alloc",
        args: [fileName, size, allocBuffer],
        callbackId: this.taskId++
      });
      return result.fileId;
    });
  }
  /**
   * Write a Blob to heapfs
   */
  fileWrite(fileId, blob) {
    return __async(this, null, function* () {
      const reader = blob.stream().getReader();
      let offset = 0;
      while (true) {
        const { done, value } = yield reader.read();
        if (done) break;
        const size = value.byteLength;
        yield this.pushTask(
          {
            verb: "fs.write",
            args: [fileId, value, offset],
            callbackId: this.taskId++
          },
          // @ts-ignore Type 'ArrayBufferLike' is not assignable to type 'ArrayBuffer'
          [value.buffer]
        );
        offset += size;
      }
    });
  }
  fileReadResponse(name, offset, size) {
    return __async(this, null, function* () {
      var _a;
      try {
        const blob = this.fileBlobs.get(name);
        if (!blob) {
          throw new Error(`blob not found for name="${name}"`);
        }
        const chunk = blob.slice(offset, offset + size);
        const buffer = yield chunk.arrayBuffer();
        this.worker.postMessage(
          { verb: "fs.read_res", args: [buffer] },
          { transfer: [buffer] }
        );
      } catch (err) {
        this.logger.error("fileReadResponse failed, terminating worker:", err);
        (_a = this.worker) == null ? void 0 : _a.terminate();
        this.worker = void 0;
        this.abort(`File read failed: ${err}`, err.stack || "");
      }
    });
  }
  /**
   * Parse JSON result returned by cpp code.
   * Throw new Error if "__exception" is present in the response
   *
   * TODO: get rid of this function once everything is migrated to Glue
   */
  parseResult(result) {
    const parsedResult = JSON.parse(result);
    if (parsedResult && parsedResult["error"]) {
      throw new WllamaRuntimeError("Unknown error, please see console.log", "");
    }
    return parsedResult;
  }
  /**
   * Push a new task to taskQueue
   */
  pushTask(param, buffers) {
    return new Promise((resolve, reject) => {
      this.taskQueue.push({ resolve, reject, param, buffers });
      this.runTaskLoop();
    });
  }
  /**
   * Main loop for processing tasks
   */
  runTaskLoop() {
    return __async(this, null, function* () {
      var _a;
      if (this.busy) {
        return;
      }
      this.busy = true;
      while (true) {
        const task = this.taskQueue.shift();
        if (!task) break;
        this.resultQueue.push(task);
        this.worker.postMessage(
          task.param,
          isSafariMobile() ? void 0 : {
            transfer: (_a = task.buffers) != null ? _a : []
          }
        );
      }
      this.busy = false;
    });
  }
  /**
   * Handle messages from worker
   */
  onRecvMsg(e) {
    if (!e.data) return;
    const { verb, args } = e.data;
    const isCompatBuild = this.resources.compat;
    if (verb && verb.startsWith("console.")) {
      if (this.suppressNativeLog) {
        return;
      }
      if (verb.endsWith("debug")) this.logger.debug(...args);
      if (verb.endsWith("log")) this.logger.log(...args);
      if (verb.endsWith("warn")) this.logger.warn(...args);
      if (verb.endsWith("error")) this.logger.error(...args);
      return;
    } else if (verb === "signal.abort") {
      const [signalType, message, rawStack, originalErr] = args;
      if (originalErr) {
        this.logger.error(originalErr);
      }
      (() => __async(this, null, function* () {
        let stack = "";
        let newMsg = message.replace(
          "Build with -sASSERTIONS for more info.",
          ""
        );
        if (signalType === "abort") {
          newMsg = `(ABORT) ${newMsg}`;
          stack = rawStack.replace(/\|/g, "\n");
        } else if (signalType === "exception") {
          stack = rawStack;
        }
        const decoded = yield Debug.decodeStackTrace(stack, isCompatBuild);
        this.logger.error(`Stack trace (${signalType}):
` + decoded);
        this.abort(newMsg, decoded);
      }))();
      return;
    }
    if (verb === FILE_READ_REQ_EVENT) {
      const [name, offset, size] = args;
      this.fileReadResponse(name, offset, size).catch(() => {
      });
      return;
    }
    const { callbackId, result, err } = e.data;
    if (callbackId) {
      const idx = this.resultQueue.findIndex(
        (t) => t.param.callbackId === callbackId
      );
      if (idx !== -1) {
        const waitingTask = this.resultQueue.splice(idx, 1)[0];
        if (err) waitingTask.reject(err);
        else waitingTask.resolve(result);
      } else {
        this.logger.error(
          `Cannot find waiting task with callbackId = ${callbackId}`
        );
      }
    }
  }
  abort(text, stack) {
    const error = new WllamaRuntimeError(
      text.length == 0 ? "(unknown error)" : text,
      stack
    );
    while (this.resultQueue.length > 0) {
      const waitingTask = this.resultQueue.pop();
      if (!waitingTask) break;
      waitingTask.reject(error);
    }
    while (this.taskQueue.length > 0) {
      const pendingTask = this.taskQueue.pop();
      if (!pendingTask) break;
      pendingTask.reject(error);
    }
  }
};

// src/huggingface.ts
var HF_BASE = "https://huggingface.co";
var DEFAULT_QUANTS = ["Q4_K_M", "Q8_0"];
function fetchRepoFiles(repo, token) {
  return __async(this, null, function* () {
    var _a;
    const url = `${HF_BASE}/api/models/${repo}/tree/main?recursive=true`;
    const headers = { Accept: "application/json" };
    if (token) {
      headers["Authorization"] = `Bearer ${token}`;
    }
    const res = yield fetch(url, { headers });
    if (!res.ok) {
      let msg = res.statusText;
      try {
        msg = (_a = (yield res.json()).error) != null ? _a : msg;
      } catch (e) {
      }
      throw new Error(`HF API error (${res.status}): ${msg}`);
    }
    return res.json();
  });
}
function firstShardPath(files, path) {
  const m = path.match(/^(.+)-(\d{5})-of-(\d{5})\.gguf$/i);
  if (!m) return path;
  const first = `${m[1]}-00001-of-${m[3]}.gguf`;
  return files.some((f) => f.path === first) ? first : path;
}
function selectFile(files, quant, mmprojOnly) {
  const candidates = files.filter((f) => {
    if (f.type !== "file" || !f.path.toLowerCase().endsWith(".gguf"))
      return false;
    const ismmproj = f.path.toLowerCase().includes("mmproj");
    return mmprojOnly ? ismmproj : !ismmproj;
  });
  if (candidates.length === 0) return null;
  if (quant) {
    const upper = quant.toUpperCase();
    const match = candidates.find((f) => f.path.toUpperCase().includes(upper));
    if (match) return firstShardPath(candidates, match.path);
    return null;
  }
  for (const q of DEFAULT_QUANTS) {
    const match = candidates.find((f) => f.path.toUpperCase().includes(q));
    if (match) return firstShardPath(candidates, match.path);
  }
  return firstShardPath(candidates, candidates[0].path);
}
function getHFModelSource(config) {
  return __async(this, null, function* () {
    const { repo, file, quant, mmprojFile, mmprojQuant, hfToken } = config;
    const files = yield fetchRepoFiles(repo, hfToken);
    const modelPath = file != null ? file : selectFile(files, quant, false);
    if (!modelPath) {
      throw new Error(`No GGUF file found in repo "${repo}"`);
    }
    const source = {
      url: `${HF_BASE}/${repo}/resolve/main/${modelPath}`
    };
    if (mmprojFile || mmprojQuant !== void 0) {
      const mmpath = mmprojFile != null ? mmprojFile : selectFile(files, mmprojQuant, true);
      if (mmpath) {
        source.mmprojUrl = `${HF_BASE}/${repo}/resolve/main/${mmpath}`;
      }
    }
    if (hfToken) {
      const params = new URLSearchParams({ token: hfToken });
      source.url += `?${params}`;
      if (source.mmprojUrl) {
        source.mmprojUrl += `?${params}`;
      }
    }
    return source;
  });
}
function getHFFileSHA256(url, headers) {
  return __async(this, null, function* () {
    if (!url.includes("/resolve/")) return void 0;
    const rawUrl = url.replace("/resolve/", "/raw/");
    try {
      const text = yield fetch(rawUrl, { headers }).then((r) => r.text());
      const match = text.match(/^oid sha256:([0-9a-f]{64})$/m);
      return match ? match[1] : void 0;
    } catch (e) {
      return void 0;
    }
  });
}

// src/storage/opfs.ts
var OPFSBackend = class {
  isSupported() {
    var _a;
    return typeof navigator !== "undefined" && "storage" in navigator && !!((_a = navigator.storage) == null ? void 0 : _a.getDirectory);
  }
  read(key) {
    return __async(this, null, function* () {
      try {
        const cacheDir = yield getCacheDir();
        const fileHandle = yield cacheDir.getFileHandle(key);
        return yield fileHandle.getFile();
      } catch (e) {
        return null;
      }
    });
  }
  write(key, stream) {
    return __async(this, null, function* () {
      const writable = yield openWritable(key);
      yield writable.truncate(0);
      const reader = stream.getReader();
      try {
        while (true) {
          const { done, value } = yield reader.read();
          if (done) break;
          yield writable.write(value);
        }
      } finally {
        yield writable.close();
      }
    });
  }
  getSize(key) {
    return __async(this, null, function* () {
      try {
        const cacheDir = yield getCacheDir();
        const fileHandle = yield cacheDir.getFileHandle(key);
        const file = yield fileHandle.getFile();
        return file.size;
      } catch (e) {
        return -1;
      }
    });
  }
  list() {
    return __async(this, null, function* () {
      const cacheDir = yield getCacheDir();
      const result = [];
      try {
        for (var iter = __forAwait(cacheDir.entries()), more, temp, error; more = !(temp = yield iter.next()).done; more = false) {
          const [name, handle] = temp.value;
          if (handle.kind === "file") {
            const file = yield handle.getFile();
            result.push({ key: name, size: file.size });
          }
        }
      } catch (temp) {
        error = [temp];
      } finally {
        try {
          more && (temp = iter.return) && (yield temp.call(iter));
        } finally {
          if (error)
            throw error[0];
        }
      }
      return result;
    });
  }
  delete(key) {
    return __async(this, null, function* () {
      try {
        const cacheDir = yield getCacheDir();
        yield cacheDir.removeEntry(key);
      } catch (e) {
        if ((e == null ? void 0 : e.name) !== "NotFoundError") throw e;
      }
    });
  }
};
function getCacheDir() {
  return __async(this, null, function* () {
    const opfsRoot = yield navigator.storage.getDirectory();
    return opfsRoot.getDirectoryHandle("cache", { create: true });
  });
}
function openWritable(fileName) {
  return __async(this, null, function* () {
    const worker = createWorker(OPFS_UTILS_WORKER_CODE);
    let pResolve;
    let pReject;
    worker.onmessage = (e) => {
      if (e.data.ok) pResolve(null);
      else if (e.data.err) pReject(e.data.err);
    };
    worker.onerror = (e) => {
      var _a;
      return pReject == null ? void 0 : pReject((_a = e.message) != null ? _a : e);
    };
    const workerExec = (data) => new Promise((resolve, reject) => {
      pResolve = resolve;
      pReject = reject;
      worker.postMessage(
        data,
        isSafariMobile() ? void 0 : { transfer: "buf" in data && data.buf ? [data.buf.buffer] : [] }
      );
    });
    yield workerExec({ action: "open", filename: fileName });
    return {
      truncate: () => __async(this, null, function* () {
      }),
      write: (value) => workerExec({ action: "write", buf: value }),
      close: () => __async(this, null, function* () {
        yield workerExec({ action: "close" });
        worker.terminate();
      })
    };
  });
}

// src/storage/cos.ts
function makeHash(key) {
  return { algorithm: "SHA-256", value: key };
}
var COSInternalBackend = class {
  isSupported() {
    return typeof navigator !== "undefined" && "crossOriginStorage" in navigator;
  }
  // IMPORTANT: key must be SHA-256 hash of the data
  read(key) {
    return __async(this, null, function* () {
      try {
        const handle = yield navigator.crossOriginStorage.requestFileHandle(
          makeHash(key)
        );
        return handle.getFile();
      } catch (e) {
        return null;
      }
    });
  }
  // IMPORTANT: key must be SHA-256 hash of the data
  write(key, stream) {
    return __async(this, null, function* () {
      const handle = yield navigator.crossOriginStorage.requestFileHandle(
        makeHash(key),
        { create: true }
      );
      const writable = yield handle.createWritable();
      const reader = stream.getReader();
      try {
        while (true) {
          const { done, value } = yield reader.read();
          if (done) break;
          yield writable.write(value);
        }
      } finally {
        yield writable.close();
      }
    });
  }
  // IMPORTANT: key must be SHA-256 hash of the data
  getSize(key) {
    return __async(this, null, function* () {
      try {
        const handle = yield navigator.crossOriginStorage.requestFileHandle(
          makeHash(key)
        );
        const file = yield handle.getFile();
        return file.size;
      } catch (e) {
        return -1;
      }
    });
  }
  list() {
    return __async(this, null, function* () {
      throw new Error("not implemented");
    });
  }
  delete(_key) {
    return __async(this, null, function* () {
      throw new Error("not implemented");
    });
  }
};
var COSBackend = class {
  constructor() {
    __publicField(this, "cos", new COSInternalBackend());
    __publicField(this, "priv", new OPFSBackend());
  }
  isSupported() {
    return this.priv.isSupported();
  }
  read(key, hint) {
    return __async(this, null, function* () {
      if ((hint == null ? void 0 : hint.sha256) && this.cos.isSupported()) {
        const blob = yield this.cos.read(hint.sha256);
        if (blob) return blob;
      }
      return this.priv.read(key);
    });
  }
  write(key, stream, hint) {
    return __async(this, null, function* () {
      if ((hint == null ? void 0 : hint.sha256) && this.cos.isSupported()) {
        yield this.cos.write(hint.sha256, stream);
      } else {
        yield this.priv.write(key, stream);
      }
    });
  }
  getSize(key, hint) {
    return __async(this, null, function* () {
      if ((hint == null ? void 0 : hint.sha256) && this.cos.isSupported()) {
        const size = yield this.cos.getSize(hint.sha256);
        if (size !== -1) return size;
      }
      return this.priv.getSize(key);
    });
  }
  list() {
    return __async(this, null, function* () {
      return this.priv.list();
    });
  }
  delete(key) {
    return __async(this, null, function* () {
      return this.priv.delete(key);
    });
  }
};

// src/cache-manager.ts
var PREFIX_METADATA = "__metadata__";
var POLYFILL_ETAG = "polyfill_for_older_version";
function hintFromMetadata(metadata) {
  if (!metadata) return void 0;
  if (metadata.sha256) return { sha256: metadata.sha256 };
  return void 0;
}
var CacheManager = class {
  /**
   * @param backends Array of storage backends to use, in order of preference ; if first is available, use it, otherwise try the next one.
   */
  constructor(backends = [new COSBackend()]) {
    __publicField(this, "sb");
    for (const backend of backends) {
      if (backend.isSupported()) {
        this.sb = backend;
        return;
      }
    }
    throw new Error("No supported storage backend found");
  }
  /**
   * Convert a given URL into a storage key.
   *
   * Format: `${hashSHA1(fullURL)}_${fileName}`
   */
  getNameFromURL(url) {
    return __async(this, null, function* () {
      return urlToFileName(url, "");
    });
  }
  /**
   * @deprecated Use `download()` instead
   *
   * Write a new file to cache. This will overwrite existing file.
   *
   * @param name The file name returned by `getNameFromURL()` or `list()`
   */
  write(name, stream, metadata) {
    return __async(this, null, function* () {
      yield this.sb.write(name, stream);
      yield this.writeMetadata(name, metadata);
    });
  }
  download(_0) {
    return __async(this, arguments, function* (url, options = {}) {
      var _a, _b, _c, _d;
      const fileKey = yield urlToFileName(url, "");
      const sha256 = yield getHFFileSHA256(url, (_a = options.headers) != null ? _a : {});
      const hint = sha256 ? { sha256 } : void 0;
      const cachedSize = yield this.sb.getSize(fileKey, hint);
      if (cachedSize !== -1) {
        const metadata2 = yield this.readMetadata(fileKey);
        if ((metadata2 == null ? void 0 : metadata2.originalURL) === url && metadata2.originalSize === cachedSize) {
          return;
        }
        const head = yield fetch(url, __spreadValues({
          method: "HEAD"
        }, options.headers ? { headers: options.headers } : {}));
        const originalSize = parseInt(
          (_b = head.headers.get("content-length")) != null ? _b : "0",
          10
        );
        const etag2 = (head.headers.get("etag") || "").replace(
          /[^A-Za-z0-9]/g,
          ""
        );
        if (originalSize > 0 && originalSize === cachedSize) {
          yield this.writeMetadata(fileKey, __spreadValues({
            originalURL: url,
            originalSize,
            etag: etag2,
            sha256
          }, (_c = options.metadataAdditional) != null ? _c : {}));
          return;
        }
        yield this.sb.delete(fileKey);
        yield this.sb.delete(`${PREFIX_METADATA}${fileKey}`);
      }
      const response = yield fetch(url, __spreadValues(__spreadValues({}, options.headers ? { headers: options.headers } : {}), options.signal ? { signal: options.signal } : {}));
      if (!response.ok || !response.body) {
        throw new Error(`Failed to fetch ${url}: HTTP ${response.status}`);
      }
      const contentLength = response.headers.get("content-length");
      const etag = (response.headers.get("etag") || "").replace(
        /[^A-Za-z0-9]/g,
        ""
      );
      const total = parseInt(contentLength != null ? contentLength : "0", 10);
      const progressCallback = options.progressCallback;
      let loaded = 0;
      let lastProgressAt = 0;
      const progressStream = new TransformStream({
        transform(chunk, controller) {
          loaded += chunk.byteLength;
          if (progressCallback) {
            const now = Date.now();
            if (now - lastProgressAt > 100) {
              lastProgressAt = now;
              progressCallback({ loaded, total });
            }
          }
          controller.enqueue(chunk);
        },
        flush() {
          progressCallback == null ? void 0 : progressCallback({ loaded, total: total || loaded });
        }
      });
      const metadata = __spreadValues({
        originalURL: url,
        originalSize: total,
        etag
      }, (_d = options.metadataAdditional) != null ? _d : {});
      if (sha256) {
        metadata.sha256 = sha256;
      }
      yield this.sb.write(
        fileKey,
        response.body.pipeThrough(progressStream),
        hint
      );
      yield this.writeMetadata(fileKey, metadata);
    });
  }
  /**
   * Open a file in cache for reading
   *
   * @param nameOrURL The file name returned by `getNameFromURL()` or `list()`, or the original URL of the remote file
   * @returns Blob, or null if file does not exist
   */
  open(nameOrURL) {
    return __async(this, null, function* () {
      const hint1 = hintFromMetadata(yield this.getMetadata(nameOrURL));
      const direct = yield this.sb.read(nameOrURL, hint1);
      if (direct) return direct;
      const key = yield urlToFileName(nameOrURL, "");
      const hint2 = hintFromMetadata(yield this.getMetadata(key));
      return this.sb.read(key, hint2);
    });
  }
  /**
   * Get the size of a file in stored cache
   *
   * NOTE: in case the download is stopped mid-way (i.e. user close browser tab), the file maybe corrupted, size maybe different from `metadata.originalSize`
   *
   * @param name The file name returned by `getNameFromURL()` or `list()`
   * @returns number of bytes, or -1 if file does not exist
   */
  getSize(name) {
    return __async(this, null, function* () {
      const hint = hintFromMetadata(yield this.getMetadata(name));
      return this.sb.getSize(name, hint);
    });
  }
  /**
   * Get metadata of a cached file
   */
  getMetadata(name) {
    return __async(this, null, function* () {
      const metadata = yield this.readMetadata(name);
      if (metadata) return metadata;
      const cachedSize = yield this.sb.getSize(name);
      return cachedSize > 0 ? (
        // files created by older version of wllama don't have metadata; polyfill it
        {
          etag: POLYFILL_ETAG,
          originalSize: cachedSize,
          originalURL: ""
        }
      ) : (
        // cached file not found
        null
      );
    });
  }
  /**
   * Same as `getMetadata()`, but without polyfill. Returns null if the file has no metadata.
   */
  readMetadata(name) {
    return __async(this, null, function* () {
      const blob = yield this.sb.read(`${PREFIX_METADATA}${name}`);
      if (!blob) return null;
      try {
        return yield new Response(blob).json();
      } catch (e) {
        return null;
      }
    });
  }
  /**
   * List all files currently in cache
   */
  list() {
    return __async(this, null, function* () {
      const all = yield this.sb.list();
      const metadataMap = {};
      for (const { key } of all) {
        if (key.startsWith(PREFIX_METADATA)) {
          const blob = yield this.sb.read(key);
          if (blob) {
            const meta = yield new Response(blob).json().catch(() => null);
            metadataMap[key.slice(PREFIX_METADATA.length)] = meta;
          }
        }
      }
      const result = [];
      for (const { key, size } of all) {
        if (!key.startsWith(PREFIX_METADATA)) {
          result.push({
            name: key,
            size,
            metadata: metadataMap[key] || {
              originalSize: size,
              originalURL: "",
              etag: ""
            }
          });
        }
      }
      return result;
    });
  }
  /**
   * Clear all files currently in cache
   */
  clear() {
    return __async(this, null, function* () {
      yield this.deleteMany(() => true);
    });
  }
  /**
   * Delete a single file in cache
   *
   * @param nameOrURL Can be either an URL or a name returned by `getNameFromURL()` or `list()`
   */
  delete(nameOrURL) {
    return __async(this, null, function* () {
      const name2 = yield this.getNameFromURL(nameOrURL);
      yield this.deleteMany(
        (entry) => entry.name === nameOrURL || entry.name === name2
      );
    });
  }
  /**
   * Delete multiple files in cache.
   *
   * @param predicate A predicate like `array.filter(item => boolean)`
   */
  deleteMany(predicate) {
    return __async(this, null, function* () {
      const list = yield this.list();
      for (const item of list) {
        if (predicate(item)) {
          yield this.sb.delete(item.name);
          yield this.sb.delete(`${PREFIX_METADATA}${item.name}`);
        }
      }
    });
  }
  /**
   * Write the metadata of the file to disk.
   */
  writeMetadata(name, metadata) {
    return __async(this, null, function* () {
      const blob = new Blob([JSON.stringify(metadata)], { type: "text/plain" });
      yield this.sb.write(`${PREFIX_METADATA}${name}`, blob.stream());
    });
  }
};
var cache_manager_default = CacheManager;
function urlToFileName(url, prefix) {
  return __async(this, null, function* () {
    const hashBuffer = yield crypto.subtle.digest(
      "SHA-1",
      new TextEncoder().encode(url)
    );
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    const hashHex = hashArray.map((b) => b.toString(16).padStart(2, "0")).join("");
    return `${prefix}${hashHex}_${url.split("/").pop()}`;
  });
}

// src/model-manager.ts
var DEFAULT_PARALLEL_DOWNLOADS = 3;
var ModelValidationStatus = /* @__PURE__ */ ((ModelValidationStatus2) => {
  ModelValidationStatus2["VALID"] = "valid";
  ModelValidationStatus2["INVALID"] = "invalid";
  ModelValidationStatus2["DELETED"] = "deleted";
  return ModelValidationStatus2;
})(ModelValidationStatus || {});
var Model = class {
  constructor(modelManager, url, mmprojUrl, savedFiles) {
    __publicField(this, "modelManager");
    /**
     * URL to the GGUF file (in case it contains multiple shards, the URL should point to the first shard)
     *
     * This URL will be used to identify the model in the cache. There can't be 2 models with the same URL.
     */
    __publicField(this, "url");
    /**
     * URL to mmproj file, if exists
     */
    __publicField(this, "mmprojUrl");
    /**
     * Size in bytes (total size of all shards).
     *
     * A value of -1 means the model is deleted from the cache. You must call `ModelManager.downloadModel` to re-download the model.
     */
    __publicField(this, "size");
    /**
     * List of all shards in the cache, sorted by original URL (ascending order)
     */
    __publicField(this, "files");
    this.modelManager = modelManager;
    this.url = url;
    this.mmprojUrl = mmprojUrl;
    if (savedFiles) {
      this.files = this.getAllFiles(savedFiles);
      this.size = sumArr(this.files.map((f) => f.metadata.originalSize));
    } else {
      this.files = [];
      this.size = 0;
    }
  }
  /**
   * Open and get a list of all shards as Blobs
   */
  open() {
    return __async(this, null, function* () {
      if (this.size === -1) {
        throw new WllamaError(
          `Model is deleted from the cache; Call ModelManager.downloadModel to re-download the model`,
          "load_error"
        );
      }
      const blobs = [];
      for (const file of this.files) {
        const blob = yield this.modelManager.cacheManager.open(file.name);
        if (!blob) {
          throw new Error(
            `Failed to open file ${file.name}; Hint: the model may be invalid, please refresh it`
          );
        }
        blobs.push(blob);
      }
      return blobs;
    });
  }
  /**
   * Validate the model files.
   *
   * If the model is invalid, the model manager will not be able to use it. You must call `refresh` to re-download the model.
   *
   * Cases that model is invalid:
   * - The model is deleted from the cache
   * - The model files are missing (or the download is interrupted)
   */
  validate() {
    let nbShards = ModelManager.parseModelUrl(this.url).length;
    if (this.mmprojUrl) {
      nbShards += 1;
    }
    if (this.size === -1) {
      return "deleted" /* DELETED */;
    }
    if (this.size < 16 || this.files.length !== nbShards) {
      return "invalid" /* INVALID */;
    }
    for (const file of this.files) {
      if (!file.metadata || file.metadata.originalSize !== file.size) {
        return "invalid" /* INVALID */;
      }
    }
    return "valid" /* VALID */;
  }
  /**
   * In case the model is invalid, call this function to re-download the model
   */
  refresh() {
    return __async(this, arguments, function* (options = {}) {
      var _a;
      const urls = ModelManager.parseModelUrl(this.url);
      if (this.mmprojUrl) {
        urls.push(this.mmprojUrl);
      }
      const works = urls.map((url, index) => ({
        url,
        index
      }));
      this.modelManager.logger.debug("Downloading model files:", urls);
      const nParallel = (_a = this.modelManager.params.parallelDownloads) != null ? _a : DEFAULT_PARALLEL_DOWNLOADS;
      const totalSize = yield this.getTotalDownloadSize(urls);
      const loadedSize = [];
      const worker = () => __async(this, null, function* () {
        while (works.length > 0) {
          const w = works.shift();
          if (!w) break;
          yield this.modelManager.cacheManager.download(w.url, __spreadProps(__spreadValues({}, options), {
            metadataAdditional: {
              originalURL: w.url,
              mmprojURL: this.mmprojUrl
            },
            progressCallback: ({ loaded }) => {
              var _a2;
              loadedSize[w.index] = loaded;
              (_a2 = options.progressCallback) == null ? void 0 : _a2.call(options, {
                loaded: sumArr(loadedSize),
                total: totalSize
              });
            }
          }));
        }
      });
      const promises = [];
      for (let i = 0; i < nParallel; i++) {
        promises.push(worker());
        loadedSize.push(0);
      }
      yield Promise.all(promises);
      this.files = this.getAllFiles(yield this.modelManager.cacheManager.list());
      this.size = this.files.reduce((acc, f) => acc + f.metadata.originalSize, 0);
    });
  }
  /**
   * Remove the model from the cache
   */
  remove() {
    return __async(this, null, function* () {
      this.files = this.getAllFiles(yield this.modelManager.cacheManager.list());
      yield this.modelManager.cacheManager.deleteMany(
        (f) => !!this.files.find((file) => file.name === f.name)
      );
      this.size = -1;
    });
  }
  getAllFiles(savedFiles) {
    const allUrls = new Set(ModelManager.parseModelUrl(this.url));
    if (this.mmprojUrl) {
      allUrls.add(this.mmprojUrl);
    }
    const allFiles = [];
    for (const url of allUrls) {
      const file = savedFiles.find((f) => f.metadata.originalURL === url);
      if (!file) {
        throw new Error(`Model file not found: ${url}`);
      }
      allFiles.push(file);
    }
    allFiles.sort(
      (a, b) => a.metadata.originalURL.localeCompare(b.metadata.originalURL)
    );
    return allFiles;
  }
  getTotalDownloadSize(urls) {
    return __async(this, null, function* () {
      const responses = yield Promise.all(
        urls.map((url) => fetch(url, { method: "HEAD" }))
      );
      const sizes = responses.map(
        (res) => Number(res.headers.get("content-length") || "0")
      );
      return sumArr(sizes);
    });
  }
};
var ModelManager = class _ModelManager {
  constructor(params = {}) {
    // The CacheManager singleton, can be accessed by user
    __publicField(this, "cacheManager");
    __publicField(this, "params");
    __publicField(this, "logger");
    this.cacheManager = params.cacheManager || new cache_manager_default();
    this.params = params;
    this.logger = params.logger || console;
  }
  /**
   * Parses a model URL and returns an array of URLs based on the following patterns:
   * - If the input URL is an array, it returns the array itself.
   * - If the input URL is a string in the `gguf-split` format, it returns an array containing the URL of each shard in ascending order.
   * - Otherwise, it returns an array containing the input URL as a single element array.
   * @param modelUrl URL or list of URLs
   */
  static parseModelUrl(modelUrl) {
    var _a;
    if (Array.isArray(modelUrl)) {
      return modelUrl;
    }
    const urlPartsRegex = /-(\d{5})-of-(\d{5})\.gguf(?:\?.*)?$/;
    const queryMatch = modelUrl.match(/\.gguf(\?.*)?$/);
    const queryParams = (_a = queryMatch == null ? void 0 : queryMatch[1]) != null ? _a : "";
    const matches = modelUrl.match(urlPartsRegex);
    if (!matches) {
      return [modelUrl];
    }
    const baseURL = modelUrl.replace(urlPartsRegex, "");
    const total = matches[2];
    const paddedShardIds = Array.from(
      { length: Number(total) },
      (_, index) => (index + 1).toString().padStart(5, "0")
    );
    return paddedShardIds.map(
      (current) => `${baseURL}-${current}-of-${total}.gguf${queryParams}`
    );
  }
  /**
   * Get all models in the cache
   */
  getModels() {
    return __async(this, arguments, function* (opts = {}) {
      const cachedFiles = yield this.cacheManager.list();
      let models = [];
      for (const file of cachedFiles) {
        if (!file.metadata.originalURL) continue;
        const shards = _ModelManager.parseModelUrl(file.metadata.originalURL);
        const mmprojUrl = file.metadata.mmprojURL;
        const isFirstShard = shards.length === 1 || shards[0] === file.metadata.originalURL;
        if (isFirstShard) {
          models.push(
            new Model(this, file.metadata.originalURL, mmprojUrl, cachedFiles)
          );
        }
      }
      if (!opts.includeInvalid) {
        models = models.filter(
          (m) => m.validate() === "valid" /* VALID */
        );
      }
      return models;
    });
  }
  /**
   * Download a model from the given URL.
   *
   * The URL must end with `.gguf`
   */
  downloadModel(_0) {
    return __async(this, arguments, function* (sourceOrURL, options = {}) {
      const source = isString(sourceOrURL) ? { url: sourceOrURL } : sourceOrURL;
      if (!isValidGgufFile(source.url)) {
        throw new WllamaError(
          `Invalid model URL: ${source.url}; URL must ends with ".gguf"`,
          "download_error"
        );
      }
      const model = new Model(this, source.url, source.mmprojUrl);
      const validity = model.validate();
      if (validity !== "valid" /* VALID */) {
        yield model.refresh(options);
      }
      return model;
    });
  }
  /**
   * Get a model from the cache or download it if it's not available.
   */
  getModelOrDownload(_0) {
    return __async(this, arguments, function* (source, options = {}) {
      var _a;
      const models = yield this.getModels();
      const model = models.find((m) => m.url === source.url);
      if (model) {
        (_a = options.progressCallback) == null ? void 0 : _a.call(options, { loaded: model.size, total: model.size });
        return model;
      }
      return this.downloadModel(source, options);
    });
  }
  /**
   * Remove all models from the cache
   */
  clear() {
    return __async(this, null, function* () {
      yield this.cacheManager.clear();
    });
  }
};

// src/types/types.ts
var LogLevel = /* @__PURE__ */ ((LogLevel2) => {
  LogLevel2[LogLevel2["DEBUG"] = 1] = "DEBUG";
  LogLevel2[LogLevel2["INFO"] = 2] = "INFO";
  LogLevel2[LogLevel2["WARN"] = 3] = "WARN";
  LogLevel2[LogLevel2["ERROR"] = 4] = "ERROR";
  return LogLevel2;
})(LogLevel || {});

// src/wasm-from-cdn.ts
var WasmCompatFromCDN = {
  worker: "https://cdn.jsdelivr.net/npm/@wllama/wllama-compat@3.12.0/wasm/wllama.js",
  wasm: "https://cdn.jsdelivr.net/npm/@wllama/wllama-compat@3.12.0/wasm/wllama.wasm"
};

// src/wllama.ts
var LoggerWithoutDebug = __spreadProps(__spreadValues({}, console), {
  debug: () => {
  }
});
var WllamaError = class extends Error {
  constructor(message, type = "unknown_error") {
    super(message);
    __publicField(this, "type");
    this.type = type;
  }
};
var WllamaAbortError = class extends Error {
  constructor() {
    super("Operation aborted");
    __publicField(this, "name", "AbortError");
  }
};
var WllamaRuntimeError = class extends Error {
  constructor(message, stack) {
    super(message);
    __publicField(this, "name", "RuntimeError");
    __publicField(this, "stack");
    this.stack = stack;
  }
};
var Wllama = class {
  constructor(pathConfig, wllamaConfig = {}) {
    // The CacheManager and ModelManager are singleton, can be accessed by user
    __publicField(this, "cacheManager");
    __publicField(this, "modelManager");
    /** Feature detection: `loraBlobs` means `lora_adapters` accepts GGUF blobs. */
    __publicField(this, "capabilities", { loraBlobs: true });
    __publicField(this, "compat", null);
    __publicField(this, "proxy", null);
    __publicField(this, "config");
    __publicField(this, "pathConfig");
    __publicField(this, "useMultiThread", false);
    __publicField(this, "nbThreads", 1);
    __publicField(this, "useEmbeddings", false);
    __publicField(this, "useRerank", false);
    // available when loaded
    __publicField(this, "loadedContextInfo", null);
    __publicField(this, "seed");
    __publicField(this, "bosToken", -1);
    __publicField(this, "eosToken", -1);
    __publicField(this, "eotToken", -1);
    __publicField(this, "eogTokens", /* @__PURE__ */ new Set());
    __publicField(this, "addBosToken", false);
    __publicField(this, "addEosToken", false);
    __publicField(this, "mediaMarker");
    __publicField(this, "chatTemplate");
    __publicField(this, "metadata");
    __publicField(this, "hasEncoder", false);
    __publicField(this, "decoderStartToken", -1);
    // note: we overlay instead of using llama-server default_template_kwargs, because we cannot transfer complex data structure via GLUE
    // overlay allow mixed data type or nested structure for kwargs
    __publicField(this, "chatTemplateKwargs", {});
    var _a, _b, _c;
    checkEnvironmentCompatible();
    if (!pathConfig) throw new WllamaError("AssetsPathConfig is required");
    this.pathConfig = pathConfig;
    this.config = wllamaConfig;
    this.cacheManager = (_a = wllamaConfig.cacheManager) != null ? _a : new cache_manager_default();
    this.modelManager = (_c = wllamaConfig.modelManager) != null ? _c : new ModelManager({
      cacheManager: this.cacheManager,
      logger: (_b = wllamaConfig.logger) != null ? _b : console,
      parallelDownloads: wllamaConfig.parallelDownloads,
      allowOffline: wllamaConfig.allowOffline
    });
    this.setCompat("default");
  }
  logger() {
    var _a;
    return (_a = this.config.logger) != null ? _a : console;
  }
  checkModelLoaded() {
    if (!this.isModelLoaded()) {
      throw new WllamaError(
        "loadModel() is not yet called",
        "model_not_loaded"
      );
    }
  }
  /**
   * Get the libllama version string, e.g. "b6327-4d74393".
   *
   * @returns version string embedded at build time.
   */
  static getLibllamaVersion() {
    return LIBLLAMA_VERSION;
  }
  /**
   * Set compatibility options for Wllama.
   * @param compat Set to null to disable compatibility, or 'default' to use the default compat resources from CDN.
   * @param mode 'safari' by default; If set to 'firefox_safari', the compat mode will **also** be enabled on Firefox, which will significantly degrade the performance but allow using WebGPU on Firefox.
   */
  setCompat(compat, mode = "safari") {
    if (mode === "safari") {
      if (isFirefox()) {
        this.compat = null;
        return;
      }
    }
    this.compat = compat === "default" ? WasmCompatFromCDN : compat;
  }
  /**
   * Check if the model is loaded via `loadModel()`
   */
  isModelLoaded() {
    return !!this.proxy && !!this.metadata;
  }
  /**
   * Get token ID associated to BOS (begin of sentence) token.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns -1 if the model is not loaded.
   */
  getBOS() {
    return this.bosToken;
  }
  /**
   * Get token ID associated to EOS (end of sentence) token.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns -1 if the model is not loaded.
   */
  getEOS() {
    return this.eosToken;
  }
  /**
   * Get token ID associated to EOT (end of turn) token.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns -1 if the model is not loaded.
   */
  getEOT() {
    return this.eotToken;
  }
  /**
   * Check if a given token is end-of-generation token (e.g. EOS, EOT, etc.)
   *
   * @param token the token ID to be checked
   * @returns true if the token is EOS, EOT, or any other end-of-generation tokens
   */
  isTokenEOG(token) {
    return token === this.eosToken || token === this.eotToken || this.eogTokens.has(token);
  }
  /**
   * Get token ID associated to token used by decoder, to start generating output sequence(only usable for encoder-decoder architecture). In other words, encoder uses normal BOS and decoder uses this token.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns -1 if the model is not loaded.
   */
  getDecoderStartToken() {
    return this.decoderStartToken;
  }
  /**
   * Get model hyper-parameters and metadata
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns ModelMetadata
   */
  getModelMetadata() {
    this.checkModelLoaded();
    return this.metadata;
  }
  /**
   * Check if we're currently using multi-thread build.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns true if multi-thread is used.
   */
  isMultithread() {
    this.checkModelLoaded();
    return this.useMultiThread;
  }
  /**
   * Get number of threads used in the current context.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns number of threads
   */
  getNumThreads() {
    this.checkModelLoaded();
    return this.useMultiThread ? this.nbThreads : 1;
  }
  /**
   * Check if the current model uses encoder-decoder architecture
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns true if multi-thread is used.
   */
  isEncoderDecoderArchitecture() {
    this.checkModelLoaded();
    return this.hasEncoder;
  }
  /**
   * Must we add BOS token to the tokenized sequence?
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns true if BOS token must be added to the sequence
   */
  mustAddBosToken() {
    this.checkModelLoaded();
    return this.addBosToken;
  }
  /**
   * Must we add EOS token to the tokenized sequence?
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns true if EOS token must be added to the sequence
   */
  mustAddEosToken() {
    this.checkModelLoaded();
    return this.addEosToken;
  }
  /**
   * Get the jinja chat template comes with the model. It only available if the original model (before converting to gguf) has the template in `tokenizer_config.json`
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns the jinja template. null if there is no template in gguf
   */
  getChatTemplate() {
    var _a;
    this.checkModelLoaded();
    return (_a = this.chatTemplate) != null ? _a : null;
  }
  /**
   * Check if WebGPU is supported by the current environment.
   * @returns true if WebGPU is supported
   */
  isSupportWebGPU() {
    return isSupportWebGPU();
  }
  /**
   * Load model from a given URL (or a list of URLs, in case the model is splitted into smaller files)
   * - If the model already been downloaded (via `downloadModel()`), then we will use the cached model
   * - Else, we download the model from internet
   * @param modelSourceOrURL
   * @param params
   */
  loadModelFromUrl(_0) {
    return __async(this, arguments, function* (modelSourceOrURL, params = {}) {
      var _a;
      const source = isString(modelSourceOrURL) ? { url: modelSourceOrURL } : modelSourceOrURL;
      const useCache = (_a = params.useCache) != null ? _a : true;
      const model = useCache ? yield this.modelManager.getModelOrDownload(source, params) : yield this.modelManager.downloadModel(source, params);
      const blobs = yield model.open();
      return yield this.loadModel(blobs, params);
    });
  }
  /**
   * Load model from a given Hugging Face model ID and file path.
   *
   * @param hfOptions
   * @param params
   */
  loadModelFromHF(_0) {
    return __async(this, arguments, function* (hfOptions, params = {}) {
      const source = yield getHFModelSource(hfOptions);
      return yield this.loadModelFromUrl(source, params);
    });
  }
  /**
   * Load model from a given list of Blob.
   *
   * You can pass multiple buffers into the function (in case the model contains multiple shards).
   *
   * @param ggufBlobsOrModel Can be either list of Blobs (in case you use local file), or a Model object (in case you use ModelManager)
   * @param params LoadModelParams
   */
  loadModel(_0) {
    return __async(this, arguments, function* (ggufBlobsOrModel, params = {}) {
      var _a, _b, _c, _d, _e, _f, _g, _h, _i, _j;
      const blobs = ggufBlobsOrModel instanceof Model ? yield ggufBlobsOrModel.open() : [...ggufBlobsOrModel];
      if (blobs.some((b) => b.size === 0)) {
        throw new WllamaError(
          "Input model (or splits) must be non-empty Blob or File",
          "load_error"
        );
      }
      if (!this.pathConfig["default"]) {
        throw new WllamaError(
          '"default" is missing from pathConfig',
          "load_error"
        );
      }
      if (this.proxy) {
        throw new WllamaError("Module is already initialized", "load_error");
      }
      const supportMultiThread = yield isSupportMultiThread();
      const hwConccurency = Math.floor((navigator.hardwareConcurrency || 1) / 2);
      const nbThreads = (_a = params.n_threads) != null ? _a : hwConccurency;
      this.nbThreads = nbThreads;
      this.useMultiThread = supportMultiThread && nbThreads > 1;
      const workerResources = this.getWorkerResources();
      if (params.n_gpu_layers === 0) {
        workerResources.noWebGPU = true;
      }
      this.proxy = new ProxyToWorker(
        workerResources,
        this.useMultiThread ? nbThreads : 0,
        // 0 means disable pthread
        (_b = this.config.suppressNativeLog) != null ? _b : false,
        this.logger()
      );
      let logLevel = (_c = params.log_level) != null ? _c : 2 /* INFO */;
      if (this.config.suppressNativeLog) {
        logLevel = 9999;
      }
      const modelFiles = yield prepareBlobs(blobs);
      const loraPaths = [];
      const loraFiles = [];
      for (const [i, adapter] of ((_d = params.lora_adapters) != null ? _d : []).entries()) {
        if (adapter.blob) {
          if (adapter.blob.size === 0) {
            throw new WllamaError(`LoRA adapter ${i} is an empty Blob`, "load_error");
          }
          const name = `lora-${padDigits(i + 1, 5)}.gguf`;
          loraFiles.push({ name, blob: adapter.blob });
          loraPaths.push(`models/${name}`);
        } else if (adapter.path) {
          loraPaths.push(adapter.path);
        } else {
          throw new WllamaError(`LoRA adapter ${i} needs a path or a blob`, "load_error");
        }
      }
      yield this.proxy.moduleInit([...modelFiles.all, ...loraFiles]);
      this.logger().debug("Calling wllamaStart...");
      const startResult = yield this.proxy.wllamaStart();
      if (!startResult.success) {
        throw new WllamaError(
          `Error while calling start function, result = ${startResult}`
        );
      }
      this.logger().debug("Loading model...");
      const loadResult = yield this.proxy.wllamaAction("load", {
        _name: "load_req",
        log_level: logLevel,
        // if async read is not supported, use mmap; refer to README-dev.md for more details
        use_mmap: !canUseAsyncFileRead(workerResources.compat),
        use_mlock: false,
        n_gpu_layers: (_e = params.n_gpu_layers) != null ? _e : 99999,
        n_ctx: (_f = params.n_ctx) != null ? _f : 1024,
        n_threads: this.useMultiThread ? nbThreads : 1,
        n_ctx_auto: false,
        // not supported for now
        mmproj_path: modelFiles.mmproj ? `/models/${MMPROJ_FILE_NAME}` : void 0,
        model_paths: modelFiles.llm.map((f) => `models/${f.name}`),
        embeddings: params.embeddings,
        offload_kqv: params.offload_kqv,
        n_batch: params.n_batch,
        n_ubatch: params.n_ubatch,
        pooling_type: params.pooling_type,
        rope_scaling_type: params.rope_scaling_type,
        rope_freq_base: params.rope_freq_base,
        rope_freq_scale: params.rope_freq_scale,
        yarn_ext_factor: params.yarn_ext_factor,
        yarn_attn_factor: params.yarn_attn_factor,
        yarn_beta_fast: params.yarn_beta_fast,
        yarn_beta_slow: params.yarn_beta_slow,
        yarn_orig_ctx: params.yarn_orig_ctx,
        cache_type_k: params.cache_type_k,
        cache_type_v: params.cache_type_v,
        // with unified KV, all sequences share one n_ctx cache, so each request can still use the full context
        n_parallel: (_g = params.n_parallel) != null ? _g : 4,
        kv_unified: (_h = params.kv_unified) != null ? _h : true,
        flash_attn: params.flash_attn,
        swa_full: params.swa_full,
        chat_template: params.chat_template,
        jinja: params.jinja,
        reasoning: params.reasoning,
        image_min_tokens: params.image_min_tokens,
        image_max_tokens: params.image_max_tokens,
        warmup: params.warmup,
        no_kv_offload: params.no_kv_offload,
        mmproj_offload: params.mmproj_offload,
        cont_batching: params.cont_batching,
        n_keep: params.n_keep,
        ctx_shift: params.ctx_shift,
        cache_idle_slots: params.cache_idle_slots,
        n_cache_reuse: params.n_cache_reuse,
        lora_paths: params.lora_adapters ? loraPaths : void 0,
        lora_scales: (_i = params.lora_adapters) == null ? void 0 : _i.map((a) => {
          var _a2;
          return (_a2 = a.scale) != null ? _a2 : 1;
        }),
        lora_init_without_apply: params.lora_init_without_apply,
        spec_draft_model: params.spec_draft_model,
        spec_draft_ngl: params.spec_draft_ngl,
        spec_draft_n_max: params.spec_draft_n_max,
        spec_draft_n_min: params.spec_draft_n_min,
        spec_draft_p_min: params.spec_draft_p_min,
        spec_draft_threads: params.spec_draft_threads,
        spec_draft_threads_batch: params.spec_draft_threads_batch,
        kv_overrides_keys: params.kv_overrides ? Object.keys(params.kv_overrides) : void 0,
        kv_overrides_vals: params.kv_overrides ? Object.values(params.kv_overrides) : void 0,
        reasoning_budget_tokens: params.reasoning_budget_tokens,
        reasoning_budget_message: params.reasoning_budget_message,
        reasoning_format: params.reasoning_format,
        skip_chat_parsing: params.skip_chat_parsing,
        prefill_assistant: params.prefill_assistant
      });
      const loadedCtxInfo = __spreadProps(__spreadValues({}, loadResult), {
        metadata: {}
      });
      for (let i = 0; i < loadResult.metadata_key.length; i++) {
        loadedCtxInfo.metadata[loadResult.metadata_key[i]] = loadResult.metadata_val[i];
      }
      this.seed = params.seed;
      this.bosToken = loadedCtxInfo.token_bos;
      this.eosToken = loadedCtxInfo.token_eos;
      this.eotToken = loadedCtxInfo.token_eot;
      this.useEmbeddings = !!params.embeddings;
      this.useRerank = params.pooling_type == "rank";
      this.metadata = {
        hparams: {
          nVocab: loadedCtxInfo.n_vocab,
          nCtxTrain: loadedCtxInfo.n_ctx_train,
          nEmbd: loadedCtxInfo.n_embd,
          nLayer: loadedCtxInfo.n_layer
        },
        meta: loadedCtxInfo.metadata
      };
      this.hasEncoder = !!loadedCtxInfo.has_encoder;
      this.decoderStartToken = loadedCtxInfo.token_decoder_start;
      this.addBosToken = loadedCtxInfo.add_bos_token;
      this.addEosToken = loadedCtxInfo.add_eos_token;
      this.chatTemplate = loadedCtxInfo.metadata["tokenizer.chat_template"];
      this.loadedContextInfo = loadedCtxInfo;
      this.eogTokens = new Set(loadedCtxInfo.list_tokens_eog);
      this.mediaMarker = loadedCtxInfo.media_marker;
      this.chatTemplateKwargs = (_j = params.default_template_kwargs) != null ? _j : {};
      this.logger().debug({ loadedCtxInfo });
    });
  }
  getLoadedContextInfo() {
    this.checkModelLoaded();
    if (!this.loadedContextInfo) {
      throw new WllamaError("Loaded context info is not available");
    }
    return __spreadValues({}, this.loadedContextInfo);
  }
  //////////////////////////////////////////////
  // High level API
  /**
   * Calculate embedding vector for a given text.
   * By default, BOS and EOS tokens will be added automatically. You can use the "skipBOS" and "skipEOS" option to disable it.
   * @param options OAI-compatible embedding creation options
   * @returns OAI-compatible embedding response
   */
  createEmbedding(options) {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      if (!this.useEmbeddings) {
        throw new WllamaError(
          "Embeddings is not enabled. Please set it via LoadModelParams.embeddings"
        );
      }
      const result = yield this.proxy.wllamaAction(
        "embedding",
        {
          _name: "embd_req",
          data_json: JSON.stringify(options),
          files: []
          // TODO: support file input
        }
      );
      if (!result.success) {
        throw new WllamaError(
          "Model failed to start inference",
          "inference_error"
        );
      }
      return yield this.getResponse(options, false, result.req_id);
    });
  }
  /**
   * Rerank a list of documents against a query.
   * Requires the model to be loaded with embeddings: true and pooling_type: 'rank'.
   * @param options Reranking options (query, documents, top_n)
   * @returns Reranking response with relevance scores sorted highest first
   */
  createRerank(options) {
    return __async(this, null, function* () {
      var _a, _b;
      this.checkModelLoaded();
      if (!this.useEmbeddings || !this.useRerank) {
        throw new WllamaError(
          "Rerank is not enabled. Please set it via LoadModelParams: embeddings = true and pooling_type = rank"
        );
      }
      const top_n = (_a = options.top_n) != null ? _a : options.documents.length;
      let totalTokens = 0;
      const rawResults = [];
      for (let i = 0; i < options.documents.length; i++) {
        const result = yield this.proxy.wllamaAction("rerank", {
          _name: "rrnk_req",
          data_json: JSON.stringify({
            query: options.query,
            document: options.documents[i]
          })
        });
        if (!result.success) {
          throw new WllamaError(
            "Model failed to start reranking",
            "inference_error"
          );
        }
        const { score, tokens_evaluated } = yield this.getSingleResult(result.req_id, "reranking");
        totalTokens += tokens_evaluated;
        rawResults.push({ index: i, score });
      }
      rawResults.sort((a, b) => b.score - a.score);
      return {
        model: (_b = this.getModelMetadata().meta["general.name"]) != null ? _b : "",
        object: "list",
        usage: { prompt_tokens: totalTokens, total_tokens: totalTokens },
        results: rawResults.slice(0, top_n).map(({ index, score }) => ({
          index,
          relevance_score: score
        }))
      };
    });
  }
  /**
   * Answer typed questions about a state, using a decision model (TypeSafe-compatible System One API).
   * Image input is not supported yet.
   * @param options The state and the questions
   * @returns The answer of each question
   */
  createSystemOne(options) {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      const result = yield this.proxy.wllamaAction(
        "systemone",
        {
          _name: "sys1_req",
          data_json: JSON.stringify(options)
        }
      );
      if (!result.success) {
        throw new WllamaError(
          "Model failed to start systemone",
          "inference_error"
        );
      }
      return yield this.getSingleResult(
        result.req_id,
        "systemone"
      );
    });
  }
  createChatCompletion(options) {
    return __async(this, null, function* () {
      var _a;
      if (Object.keys(this.chatTemplateKwargs).length > 0) {
        options = __spreadProps(__spreadValues({}, options), {
          chat_template_kwargs: __spreadValues(__spreadValues({}, this.chatTemplateKwargs), (_a = options.chat_template_kwargs) != null ? _a : {})
        });
      }
      if (options.stream && options.onData) {
        yield this.createCompletionImpl(options);
      } else if (options.stream) {
        return yield this.createCompletionGenerator(options);
      } else {
        return yield this.createCompletionImpl(__spreadProps(__spreadValues({}, options), { stream: false }));
      }
    });
  }
  createCompletion(options) {
    return __async(this, null, function* () {
      if (options.stream && options.onData) {
        yield this.createCompletionImpl(options);
      } else if (options.stream) {
        return yield this.createCompletionGenerator(options);
      } else {
        return yield this.createCompletionImpl(__spreadProps(__spreadValues({}, options), { stream: false }));
      }
    });
  }
  /**
   * Private implementation of createCompletion
   */
  createCompletionImpl(options) {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      const isStream = !!options.stream;
      const isChat = !!options.messages;
      const customOpt = {};
      if (this.seed !== void 0) {
        customOpt.seed = this.seed;
      }
      let files = [];
      if (isChat) {
        const tmp = this.prepareMultimodalInput(
          options
        );
        options = tmp.params;
        files = tmp.files;
      }
      const result = yield this.proxy.wllamaAction(
        "completion",
        {
          _name: "cmpl_req",
          is_chat: isChat,
          data_json: JSON.stringify(__spreadValues(__spreadValues({}, options), customOpt)),
          files: files.map((f) => new Uint8Array(f))
        }
      );
      if (!result.success) {
        throw new WllamaError(
          "Model failed to start inference",
          "inference_error"
        );
      }
      return yield this.getResponse(
        options,
        isStream,
        result.req_id
      );
    });
  }
  /**
   * Same with `createCompletion`, but returns an async iterator instead.
   * Only called when stream=true and no onData is provided.
   */
  createCompletionGenerator(options) {
    return new Promise((resolve) => {
      const createGenerator = cbToAsyncIter(
        (callback) => {
          this.createCompletionImpl(__spreadProps(__spreadValues({}, options), {
            onData: (chunk) => callback(chunk)
          })).then(() => callback(void 0, true)).catch((err) => callback(void 0, false, err));
        }
      );
      resolve(createGenerator());
    });
  }
  /**
   * Whether the currently loaded model supports a specific input modality (e.g. image or audio).
   * @param modality
   * @returns
   */
  supportInputModality(modality) {
    this.checkModelLoaded();
    if (modality === "image") {
      return !!this.loadedContextInfo.has_image_input;
    } else if (modality === "audio") {
      return !!this.loadedContextInfo.has_audio_input;
    } else {
      throw new WllamaError(
        "Unsupported modality: " + modality,
        "unknown_error"
      );
    }
  }
  /**
   * Unload the model and free all memory.
   *
   * Note: This function will NOT crash if model is not yet loaded
   */
  exit() {
    return __async(this, null, function* () {
      var _a;
      yield (_a = this.proxy) == null ? void 0 : _a.wllamaExit();
      this.proxy = null;
    });
  }
  /**
   * [FOR DEBUGGING ONLY] Run ggml backend ops tests without loading any model.
   *
   * Initializes the wasm runtime, executes `test-backend-ops` with the given args, then shuts down.
   *
   * For more info, please refer to guides/debug.md
   *
   * @param args Arguments forwarded to test-backend-ops (e.g. ["-o", "ADD"])
   * @returns retcode (0 = all tests passed) and success flag
   */
  testBackendOps() {
    return __async(this, arguments, function* (args = []) {
      var _a;
      if (!this.pathConfig["default"]) {
        throw new WllamaError(
          '"default" is missing from pathConfig',
          "load_error"
        );
      }
      if (!(yield isSupportMultiThread())) {
        throw new WllamaError(
          "Multi-threading is required to run backend ops tests, but it is not supported in the current environment."
        );
      }
      const tmpProxy = new ProxyToWorker(
        this.getWorkerResources(),
        0,
        // single-thread; no model needed
        (_a = this.config.suppressNativeLog) != null ? _a : false,
        this.logger()
      );
      try {
        yield tmpProxy.moduleInit([]);
        const startResult = yield tmpProxy.wllamaStart();
        if (!startResult.success) {
          throw new WllamaError(
            `Error while calling start function, result = ${startResult}`
          );
        }
        const result = yield tmpProxy.wllamaAction(
          "test_backend_ops",
          { _name: "tbop_req", args: ["test-backend-ops", ...args] }
        );
        return { retcode: result.retcode, success: result.success };
      } finally {
        yield tmpProxy.wllamaExit();
      }
    });
  }
  //////////////////////////////////////////////
  // Low level API
  // TODO: add back
  /**
   * get debug info
   */
  _getDebugInfo() {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      return yield this.proxy.wllamaDebug();
    });
  }
  //////////////////////////////////////////////
  // Utils
  jsonDecode(data_json) {
    try {
      return JSON.parse(data_json);
    } catch (e) {
      this.logger().error("Failed to parse JSON:", data_json);
      throw new WllamaError("Failed to parse model output", "inference_error");
    }
  }
  prepareMultimodalInput(params) {
    const msg = params.messages;
    const msgNew = [];
    const files = [];
    for (const m of msg) {
      if (Array.isArray(m.content)) {
        const newContent = [];
        for (const c of m.content) {
          if (c.type === "text") {
            newContent.push(c);
          } else {
            if (!this.mediaMarker) {
              throw new WllamaError(
                "Media marker is undefined",
                "inference_error"
              );
            }
            files.push(c.data);
            newContent.push({
              type: "text",
              text: this.mediaMarker
            });
          }
        }
        msgNew.push(__spreadProps(__spreadValues({}, m), {
          content: newContent
        }));
      } else {
        msgNew.push(m);
      }
    }
    return {
      params: __spreadProps(__spreadValues({}, params), {
        messages: msgNew
      }),
      files
    };
  }
  // release the slot occupied by the request; cancelling an already-finished request is a no-op
  cancelRequest(reqId) {
    return __async(this, null, function* () {
      try {
        yield this.proxy.wllamaAction("cancel", {
          _name: "cncl_req",
          req_id: reqId
        });
      } catch (e) {
        this.logger().warn("Failed to cancel request", reqId, e);
      }
    });
  }
  getSingleResult(reqId, task) {
    return __async(this, null, function* () {
      let completed = false;
      try {
        while (true) {
          const chunk = yield this.proxy.wllamaAction(
            "get_result",
            { _name: "gres_req", req_id: reqId }
          );
          const jsonString = chunk.data_json;
          if (jsonString && jsonString.length > 0) {
            if (chunk.is_error) {
              const jsonData = this.jsonDecode(jsonString);
              throw new WllamaError(
                jsonData.message || `Unknown ${task} error`,
                "inference_error"
              );
            }
            completed = true;
            return this.jsonDecode(jsonString);
          }
          if (!chunk.has_more) {
            completed = true;
            break;
          }
        }
        throw new WllamaError(`No ${task} result received`, "inference_error");
      } finally {
        if (!completed) {
          yield this.cancelRequest(reqId);
        }
      }
    });
  }
  getResponse(options, isStream, reqId) {
    return __async(this, null, function* () {
      var _a, _b;
      let finalResult = null;
      let completed = false;
      try {
        while (true) {
          if ((_a = options.abortSignal) == null ? void 0 : _a.aborted) {
            throw new WllamaAbortError();
          }
          const result_chunk = yield this.proxy.wllamaAction(
            "get_result",
            {
              _name: "gres_req",
              req_id: reqId
            }
          );
          const jsonString = result_chunk.data_json;
          if (!jsonString || jsonString.length === 0) {
            if (!result_chunk.has_more) {
              completed = true;
              break;
            } else {
              continue;
            }
          }
          if (jsonString == "null") {
            continue;
          }
          let jsonData = this.jsonDecode(jsonString);
          finalResult = jsonData;
          if (result_chunk.is_error) {
            this.logger().error("Model returned an error:", jsonData);
            throw new WllamaError(
              jsonData.message || "Unknown inference error",
              "inference_error"
            );
          }
          if (isStream) {
            if (!Array.isArray(jsonData)) {
              jsonData = [jsonData];
            }
            for (const chunk of jsonData) {
              (_b = options.onData) == null ? void 0 : _b.call(options, chunk);
              finalResult = chunk;
            }
          }
          if (!result_chunk.has_more) {
            completed = true;
            break;
          }
        }
      } finally {
        if (!completed) {
          yield this.cancelRequest(reqId);
        }
      }
      return finalResult;
    });
  }
  getWorkerResources() {
    const workerResources = {
      wasmPath: absoluteUrl(this.pathConfig["default"]),
      compat: false
    };
    if (needCompat()) {
      if (!this.compat) {
        this.logger().warn(
          "Not using compat mode" + (isFirefox() ? " (expected on Firefox - WebGPU will be disabled)" : "")
        );
      } else {
        const isUsingDefault = this.compat.worker === WasmCompatFromCDN.worker && this.compat.wasm === WasmCompatFromCDN.wasm;
        if (isUsingDefault) {
          this.logger().warn(
            "Compatibility mode is activated, using resources from CDN. To use local resources, please refer to @wllama/wllama-compat package."
          );
          this.logger().warn(
            "IMPORTANT: Performance will be significantly degraded in compatibility mode."
          );
        }
        workerResources.wasmPath = absoluteUrl(this.compat.wasm);
        workerResources.jsPath = this.compat.worker;
        workerResources.compat = true;
      }
    }
    if (isFirefox()) {
      if (workerResources.compat) {
        this.logger().warn(
          'Using compat mode on Firefox, performance will be significantly degraded; Consider enabling "javascript.options.wasm_js_promise_integration" in "about:config".'
        );
      } else if (!isSupportJSPI()) {
        this.logger().warn(
          'WebGPU is disabled on Firefox due to missing JSPI support. Please consider enabling compat mode, or enabling "javascript.options.wasm_js_promise_integration" in "about:config".'
        );
      }
    }
    return workerResources;
  }
};
export {
  CacheManager,
  LogLevel,
  LoggerWithoutDebug,
  Model,
  ModelManager,
  ModelValidationStatus,
  POLYFILL_ETAG,
  Wllama,
  WllamaAbortError,
  WllamaError,
  WllamaRuntimeError,
  getHFFileSHA256,
  getHFModelSource,
  isValidGgufFile
};
